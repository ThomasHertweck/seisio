"""I/O of seismic files in SU format."""

import fsspec
import json
import logging
import numpy as np

from pathlib import Path

from . import reader
from . import tools
from . import writer

log = logging.getLogger(__name__)

_SUHEADSIZE = 240


class Reader(reader.Reader):
    """Class to deal with input of seismic files in SU format."""

    def __init__(self, file, **kwargs):
        """
        Initialize class Reader.

        Parameters
        ----------
        file : str
            The name of the SU input file to read.
        storage_options : dict (default: None)
            Storage options to pass to the storage backend (if remote).
        endian : char, optional (default: None)
            Endianess of the input file, ">" for big endian, "<" for little
            endian. If endian=None (default), then the endianess of the file
            is determined automatically. This parameter can be used to force
            a specific setting.
        thdef : str, optional
            The name of the SU trace header definition JSON file. Defaults
            to the standard SU trace header definition as originally provided
            by the SU segy.h source code file.
        mnemonic_dt : str, optional (default: "dt")
            The SU trace header mnemonic specifying the sampling interval.
            Using this parameter is only useful in case the standard trace
            header definition is changed.
        mnemonic_ns : str, optional (default: "ns")
            The SU trace header mnemonic specifying the number of samples.
            Using this parameter is only useful in case the standard trace
            header definition is changed.
        mnemonic_delrt : str, optional (default: "delrt")
            The SU trace header mnemonic specifying the delay recording time.
            Using this parameter is only useful in case the standard trace
            header definition is changed.
        mnemonic_scalco : str, optional (default: "scalco")
            The SU trace header mnemonic specifying the coordinate scaler.
        mnemonic_scalel : str, optional (default: "scalel")
            The SU trace header mnemonic specifying the elevation scaler.
        """
        super().__init__(file, **kwargs)

        self._fp.endian = kwargs.pop("endian", None)

        self._par["thdef"] = kwargs.pop("thdef", None)
        self._par["mnemonic_dt"] = kwargs.pop("mnemonic_dt", "dt")
        self._par["mnemonic_ns"] = kwargs.pop("mnemonic_ns", "ns")
        self._par["mnemonic_delrt"] = kwargs.pop("mnemonic_delrt", "delrt")
        self._par["mnemonic_scalco"] = kwargs.pop("mnemonic_scalco", "scalco")
        self._par["mnemonic_scalel"] = kwargs.pop("mnemonic_scalel", "scalel")

        if kwargs:
            for key, val in kwargs.items():
                if key != "storage_options":
                    log.warning("Unknown argument '%s' with value '%s' encountered.", key, str(val))

        self._fp.mode = "r"
        self._fp.datfmt = 5
        self._fp.fixed = True
        self._fp.skip = 0
        self._fp.dtype = np.dtype(tools._DATAFORMAT[self._fp.datfmt]['dtype'])

        if self._fp.endian is not None and self._fp.endian not in ["<", ">"]:
            raise ValueError(f"Unknown value '{self._fp.endian}' for argument 'endian'.")

        if self._par["thdef"] is None:
            self._par["thdef"] = Path(__file__).parent/"json/su_traceheader.json"

        with open(self._par["thdef"]) as io:
            self._tr.thdict = json.load(io)

        log.info("Assuming fixed-length traces for SU data.")
        log.info("Data sample format: %s.", tools._DATAFORMAT[self._fp.datfmt]["desc"])

        if self._fp.endian is None:
            self._fp.endian = self._guess_endianess()
            log.info("Input file endianess looks to be '%s' (best guess).", self._fp.endian)
        else:
            log.info("Input file endianess set to '%s'.", self._fp.endian)

        log.info("Byte offset of first trace relative to start of file: %d bytes.",
                 self._fp.skip)

        self._get_fileattr()

    def _guess_endianess(self):
        """
        Guess endianess of an SU file.

        Returns
        -------
        char : endianess, either "<" (little), ">" (big) or "=" (native)
        """
        byte_ns = self._tr.thdict[self._par["mnemonic_ns"]]["byte"]
        type_ns = self._tr.thdict[self._par["mnemonic_ns"]]["type"]
        byte_dt = self._tr.thdict[self._par["mnemonic_dt"]]["byte"]
        type_dt = self._tr.thdict[self._par["mnemonic_dt"]]["type"]

        with self._fp.fs.open(self._fp.uri, "rb") as fio:
            dtp_le = np.dtype(f"<{type_ns}")
            dtp_be = np.dtype(f">{type_ns}")
            ns_bytes = fsspec.utils.read_block(fio, offset=byte_ns-1, length=dtp_le.itemsize)
            ns_le = np.frombuffer(ns_bytes, dtype=dtp_le, count=1)[0]
            ns_be = np.frombuffer(ns_bytes, dtype=dtp_be, count=1)[0]
            dtp_le = np.dtype(f"<{type_dt}")
            dtp_be = np.dtype(f">{type_dt}")
            dt_bytes = fsspec.utils.read_block(fio, offset=byte_dt-1, length=dtp_le.itemsize)
            dt_le = np.frombuffer(dt_bytes, dtype=dtp_le, count=1)[0]
            dt_be = np.frombuffer(dt_bytes, dtype=dtp_be, count=1)[0]

        if (dt_be <= 10000) and (ns_be <= 20000):  # likely big endian
            return '>'
        if (dt_le <= 10000) and (ns_le <= 20000):  # likely little endian
            return '<'
        return '='  # unsure; go with native endian

    def _get_fileattr(self):
        """Determine certain attributes by analzying file."""
        keys, formats, titles = tools._parse_hdef(self._tr.thdict, endian=self._fp.endian)
        self._tr.thdtype = tools._create_dtype(keys, formats, titles=titles)
        self._tr.thsize = self._tr.thdtype.itemsize
        if self._tr.thsize != _SUHEADSIZE:
            log.warning("Size of trace header (%d bytes) violates SU standard (%d bytes)",
                        self._tr.thsize, _SUHEADSIZE)

        with self._fp.fs.open(self._fp.uri, "rb") as fio:
            header_bytes = fsspec.utils.read_block(fio, offset=self._fp.skip,
                                                   length=self._tr.thdtype.itemsize)
            headers = np.frombuffer(header_bytes, dtype=self._tr.thdtype, count=1)
            self._dp.ns = headers[self._par["mnemonic_ns"]][0]
            self._dp.si = headers[self._par["mnemonic_dt"]][0]
            self._dp.delay = headers[self._par["mnemonic_delrt"]][0]
            self._dp.scalco = headers[self._par["mnemonic_scalco"]][0]
            self._dp.scalel = headers[self._par["mnemonic_scalel"]][0]
            fio.seek(0, 2)
            self._fp.filesize = fio.tell()

        keys.append("data")
        formats.append(f"({self._dp.ns},){self._fp.endian}"
                       f"{tools._DATAFORMAT[self._fp.datfmt]['type']}")
        titles.append("Seismic data")
        self._tr.trdtype = tools._create_dtype(keys, formats, titles=titles)
        self._tr.trsize = self._tr.trdtype.itemsize

        self._dp.nt = np.int64(self.fsize/self.trsize)
        if self.fsize % self.trsize != 0:
            log.warning("Length mismatch encountered in file %s; trying to continue.",
                        self._fp.file)
            log.warning("Filesize: %d bytes, trace size: %d bytes.",
                        self.fsize, self.trsize)

        log.info("Number of samples per data trace: %d.", self.ns)
        log.info("Sampling interval: %s (unit as per SU standard).", str(self.vsi))
        log.info("Delay (on first trace): %s (unit as per SU standard).", str(self._dp.delay))
        log.info("Coordinate scaler (on first trace): %s.", str(self._dp.scalco))
        log.info("Elevation scaler (on first trace): %s.", str(self._dp.scalel))
        log.info("Number of data traces in file: %d.", self.nt)

class Writer(writer.Writer):
    """Class to deal with ouput of seismic files in SU format."""

    def __init__(self, file, ns=None, mode="w", **kwargs):
        """
        Initialize class Writer.

        Parameters
        ----------
        file : str or pathlib.Path
            The name of the SU output file to write.
        ns : int
            Number of samples per output trace.
        storage_options : dict (default: None)
           Storage options to pass to the storage backend (if remote).
        mode : char, optional (default: "w")
            File opening mode. "w" writes a new file, or truncates an existing
            file if it already exists. "a" writes a new file, or appends to an
            existing file if it already exists.
        endian : char, optional (default: "=")
            Endianess of the input file, ">" for big endian, "<" for little
            endian, "=" for native endian.
        thdef : str, optional
            The name of the SU trace header definition JSON file. Defaults to
            the standard SU trace header definition as originally provided by
            the SU segy.h source code file.
        """
        super().__init__(file, mode, **kwargs)

        self._dp.ns = ns
        self._fp.endian = kwargs.pop("endian", "=")
        self._par["thdef"] = kwargs.pop("thdef", None)

        if kwargs:
            for key, val in kwargs.items():
                log.warning("Unknown argument '%s' with value '%s' encountered.", key, str(val))

        self._fp.datfmt = 5
        self._fp.fixed = True
        self._head_written = True
        self._tail_written = False

        if self._dp.ns is None or self._dp.ns <= 0:
            raise ValueError("Need number of output samples, argument 'ns', greater than zero.")

        if self._fp.endian not in ["<", ">", "="]:
            raise ValueError(f"Unknown value '{self._fp.endian}' for argument 'endian'.")

        if self._fp.endian == "=":
            self._fp.endian = tools._native_endian()

        self._fp.dtype = np.dtype(tools._DATAFORMAT[self._fp.datfmt]['dtype'])

        if self._par["thdef"] is None:
            self._par["thdef"] = Path(__file__).parent/"json/su_traceheader.json"

        with open(self._par["thdef"]) as io:
            self._tr.thdict = json.load(io)

        keys, formats, titles = tools._parse_hdef(self._tr.thdict, endian=self._fp.endian)
        self._tr.thdtype = tools._create_dtype(keys, formats, titles=titles)
        self._tr.thsize = self._tr.thdtype.itemsize

        keys.append("data")
        formats.append(f"({self._dp.ns},){self._fp.endian}"
                       f"{tools._DATAFORMAT[self._fp.datfmt]['type']}")
        titles.append("Seismic data")
        self._tr.trdtype = tools._create_dtype(keys, formats, titles=titles)
        self._tr.trsize = self._tr.trdtype.itemsize

        log.info("Assuming fixed-length traces for SU data.")
        log.info("Output number of samples per data trace: %d.", self.ns)
        log.info("Output data sample format: %s.", tools._DATAFORMAT[self._fp.datfmt]["desc"])
        log.info("Output file endianess set to '%s'.", self._fp.endian)
        log.info("Byte offset of first trace relative to start of file: %d bytes.",
                 self._fp.skip)
