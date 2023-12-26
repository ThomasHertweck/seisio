"""I/O of seismic files in SEG2 format."""

import logging
import numpy as np
import pandas as pd
import time

from struct import unpack_from

from . import seisio
from . import tools

log = logging.getLogger(__name__)


class Reader(seisio.SeisIO):
    """Class to deal with input of seismic files in SEG2 format."""

    def __init__(self, file):
        """
        Initialize class Reader.

        Parameters
        ----------
        file : str or pathlib.Path
            The name of the SEG2 input file to read.
        """
        super().__init__(file)

        self._sg2 = self._SEG2()
        self._fp.fixed = True

        log.info("Input file: %s", self._fp.file)

        with open(self._fp.file, "rb") as fio:
            fio.seek(0, 2)
            self._fp.filesize = fio.tell()
            fio.seek(0, 0)

            # file descriptor block and free format section
            tpb_size = self._read_file_descriptor_block(fio)
            log.debug("Size of file descriptor block: %d bytes.", tpb_size)
            self._read_trace_pointer_subblock(fio, tpb_size)
            free_form_str = fio.read(self._sg2.trcptr[0]-fio.tell())
            self._sg2.fheader = self._parse_free_format_section(self._sg2.fheader, free_form_str)
            self._check_fdb_strings()
            self._fp.skip = self._sg2.trcptr[0]
            # first trace header descriptor block
            fio.seek(self._sg2.trcptr[0], 0)
            tdb_size = self._read_trace_descriptor_block(fio)
            log.debug("Size of first trace descriptor block: %d bytes.", tpb_size)
            free_form_str = fio.read(tdb_size-32)
            self._sg2.theader = self._parse_free_format_section(self._sg2.theader, free_form_str)
            self._check_tdb_strings()

    @property
    def fheader(self):
        """
        Get information from the free-form file header.

        Returns
        -------
        dict
            The file header as dictionary.
        """
        return self._sg2.fheader

    @property
    def vsi(self):
        """
        Get the (vertical) sampling interval (of first trace).

        Returns
        -------
        float
            Sampling interval.
        """
        return self._dp.si

    def _read_file_descriptor_block(self, fio):
        """Read SEG2 file descriptor block."""
        fdb = fio.read(32)

        # Determine the endianness and check validity of block
        magic = np.frombuffer(fdb, dtype='B', count=2)
        if magic[0] == 85 and magic[1] == 58:
            self._fp.endian = "<"
        elif magic[0] == 58 and magic[1] == 85:
            self._fp.endian = ">"
        else:
            log.error("File descriptor block does not start 0x3a55 or 0x553a.")
            raise ValueError("Invalid file descriptor block magic number.")
        log.info("Valid SEG2 file descriptor found. File endianess: '%s'.", self._fp.endian)

        # Check revision number
        revno = np.frombuffer(fdb, dtype=f"{self._fp.endian}H", count=1, offset=2)
        if revno != 1:
            log.warning("Only SEG2 rev. 1 is supported. File's revision is "
                        "%d, though. Trying to continue.", revno)
        else:
            log.info("File standard is SEG2 rev. 1.")

        # Determine size of Trace Pointer Subblock in bytes
        tpb_size, ntraces = np.frombuffer(fdb, dtype=f"{self._fp.endian}H", count=2, offset=4)
        log.debug("Size of trace pointer subblock: %d bytes.", tpb_size)

        if ntraces*4 > tpb_size:
            log.error("File indicates %d traces but there are only %d trace "
                      "pointers.", ntraces, tpb_size//4)
            raise ValueError("Mismatch of number of traces and trace pointer block size.")
        self._dp.nt = ntraces
        log.info("Number of data traces in file: %d.", self.nt)

        # Define string and line terminator
        st_size, fstchar, sstchar, lt_size, fltchar, sltchar = unpack_from("BccBcc", fdb, 8)

        log.debug("String terminator size: %d, characters '%s','%s'",
                  st_size, fstchar, sstchar)
        log.debug("Line terminator size: %d, characters '%s','%s'",
                  lt_size, fltchar, sltchar)

        # Assemble the string terminator
        if st_size == 1:
            self._sg2.sterm = fstchar
        elif st_size == 2:
            self._sg2.sterm = fstchar + sstchar
        else:
            raise ValueError("String terminator has wrong size.")

        # Assemble the line terminator
        if lt_size == 1:
            self._sg2.lterm = fltchar
        elif lt_size == 2:
            self._sg2.lterm = fltchar + sltchar
        else:
            raise ValueError("Line terminator has wrong size.")

        return tpb_size

    def _read_trace_pointer_subblock(self, fio, tpb_size):
        """
        Read the Trace Pointer Subblock.

        The Trace Pointer Subblock starts at byte 32, and contains pointers
        (unsigned long integers) to the start of each Trace Descriptor Block
        contained in the file. The length of this subblock in bytes (M) is
        specified in bytes 4 and 5, and the number of pointers contained in
        the subblock (N) is specified in bytes 6 and 7 (see above). This
        fixed format allows a file to be built up one trace at a time by
        upgrading the value N and adding a pointer to the new trace in the
        Trace Pointer Subblock, but without otherwise changing the file format.
        """
        fio.seek(32, 0)
        tpb = fio.read(tpb_size)
        self._sg2.trcptr = np.array(unpack_from(self._fp.endian+("L"*self.nt), tpb), dtype="int64")

    def _parse_free_format_section(self, header, free_form_str):
        """
        Parse the free form section.

        Parameter
        ---------
        header : dict
            The header dictionary to update.
        free_form_str : str
            The free-form string to parse.
        """
        def cleanup_and_decode_string(value):
            def is_good_char(c):
                return c in (b'0123456789'
                             b'abcdefghijklmnopqrstuvwxyz'
                             b'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                             b'!"#$%&\'()*+,-./:; <=>?@[\\]^_`{|}~ ')
            return "".join(map(chr, filter(is_good_char, value))).strip()

        offset = 0
        strings = []
        while offset+2 < len(free_form_str):
            strlen = np.frombuffer(free_form_str, dtype=f"{self._fp.endian}H",
                                   count=1, offset=offset)[0]
            if strlen == 0:
                break
            curstr = free_form_str[offset+2:offset+strlen]
            try:
                curstrlen = curstr.index(self._sg2.sterm)
            except ValueError:
                strings.append(curstr)
            else:
                strings.append(curstr[:curstrlen])
            offset += strlen

        for string in strings:
            string = string.strip().split(b" ", 1)
            key = cleanup_and_decode_string(string[0])
            try:
                value = string[1]
            except IndexError:
                value = ""
            if key == "NOTE":
                if len(value) > 0:
                    value = [cleanup_and_decode_string(line)
                             for line in value.split(self._sg2.lterm) if line]
                else:
                    value = ""
            else:
                value = cleanup_and_decode_string(value)
            header[key] = value

        return header

    def _check_fdb_strings(self):
        """Check strings in file descriptor block."""
        for key in self._sg2.fheader:
            if key not in tools._SEG2FILEDESCSTR:
                log.warning("Unknown string '%s' found in free format section.", key)
                log.warning("Value associated with this key: %s.", self._sg2.fheader[key])

    def _read_trace_descriptor_block(self, fio):
        """
        Read first part of a trace descriptor block.

        The file pointer must have been positioned correctly.
        """
        tdb = fio.read(32)
        # Check validity of block
        magic = np.frombuffer(tdb, dtype=f"{self._fp.endian}H", count=1)
        if magic != 17442:
            log.error("Trace descriptor block does not start 0x4422.")
            raise ValueError("Invalid trace descriptor block magic number.")
        else:
            log.info("Valid SEG2 trace descriptor found on first trace.")

        self._tr.thsize = np.frombuffer(tdb, dtype=f"{self._fp.endian}H", count=1, offset=2)[0]
        trdsize = np.frombuffer(tdb, dtype=f"{self._fp.endian}I", count=1, offset=4)[0]
        self._tr.trsize = self._tr.thsize + trdsize
        self._dp.ns = np.frombuffer(tdb, dtype=f"{self._fp.endian}I", count=1, offset=8)[0]
        self._fp.datfmt = np.frombuffer(tdb, dtype=f"{self._fp.endian}H", count=1, offset=12)[0]

        if self._fp.datfmt not in tools._SEG2DATAFORMAT:
            raise ValueError(f"SEG2 data format '{self._fp.datfmt}' not supported.")
        log.info("Number of samples on first data trace: %d.", self.ns)
        log.info("Data sample format: %s.", tools._SEG2DATAFORMAT[self._fp.datfmt]["desc"])

        return self._tr.thsize

    def _check_tdb_strings(self):
        """Check strings in (first) trace descriptor block."""
        for i, key in enumerate(tools._SEG2TRACEDESCSTR):
            self._tr.thdict[key] = {"byte": i+1, "type": "str",
                                    "desc": tools._SEG2TRACEDESCALIAS[i]}

        i = len(tools._SEG2TRACEDESCSTR)
        for key in self._sg2.theader:
            if (key not in tools._SEG2TRACEDESCSTR):
                i += 1
                log.warning("Unknown string '%s' found in trace descriptor block of 1st trace.", key)
                log.warning("Adding string to trace header table.")
                self._tr.thdict[key] = {"byte": i, "type": "str",
                                        "desc": "...non-standard SEG2 string..."}
        self._tr.thdict["NSAMPLES"] = {"byte": i+1, "type": "str", "desc": "Number of samples"}
        self._dp.si = float(self._sg2.theader[tools._SEG2TRACEDESCSTR[25]])

    def read_dataset(self, silent=False):
        """Get all traces - an alias for read_all_traces()."""
        return self.read_all_traces(silent=silent)

    def read_all_traces(self, silent=False):
        """
        Get all traces (e.g., read the entire file).

        Parameters
        ----------
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).

        Returns
        -------
        Numpy array, Pandas dataframe
            Data and trace headers.
        """
        if not silent:
            log.info("Reading entire data set (%d traces) from disk...", self.nt)

        st = time.time()
        all_headers = []
        all_data = []

        with open(self._fp.file, "rb") as fio:
            for trace in np.arange(self.nt):
                trp = self._sg2.trcptr[trace]
                fio.seek(trp, 0)
                d, h = self._parse_trace(fio, trace)
                all_headers.append(h)
                all_data.append(d)

        max_ns = self.ns
        if not self._fp.fixed:
            for d in all_data:
                max_ns = len(d) if len(d) > max_ns else max_ns
            if max_ns != self.ns:
                self._dp.ns = max_ns
                log.warning("Updated number of samples to maximum in data set: %d.", self.ns)

        data = np.zeros((self.nt, max_ns), dtype=tools._SEG2DATAFORMAT[self._fp.datfmt]["dtype"])
        df = pd.DataFrame(columns=self.mnemonics, index=range(self.nt))

        if not self._fp.fixed:
            for i, d in enumerate(all_data):
                if len(d) < max_ns:
                    data[i, 0:len(d)] = all_data[i]
                else:
                    data[i, :] = all_data[i]
                for key in self.mnemonics:
                    df.at[i, key] = all_headers[i].get(key, 0)
        else:
            for i in np.arange(self.nt):
                data[i, :] = all_data[i]
                for key in self.mnemonics:
                    df.at[i, key] = all_headers[i].get(key, 0)

        et = time.time()
        if not silent:
            diff = et-st
            if diff < 0.1:
                log.info("Reading all traces took %.3f seconds.", et-st)
            else:
                log.info("Reading all traces took %.1f seconds.", et-st)

        return data, df

    def read_all_headers(self, silent=False):
        """
        Get all trace headers.

        Parameters
        ----------
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).

        Returns
        -------
        Pandas dataframe
            Trace header table.
        """
        if not silent:
            log.info("Reading all (i.e., %d) trace headers from disk...", self.nt)

        st = time.time()
        all_headers = []

        with open(self._fp.file, "rb") as fio:
            for trace in np.arange(self.nt):
                trp = self._sg2.trcptr[trace]
                fio.seek(trp, 0)
                _, h = self._parse_trace(fio, trace, headers_only=True)
                all_headers.append(h)

        df = pd.DataFrame(columns=self.mnemonics, index=range(self.nt))

        for i in np.arange(len(all_headers)):
            for key in self.mnemonics:
                df.at[i, key] = all_headers[i].get(key, 0)

        et = time.time()
        if not silent:
            diff = et-st
            if diff < 0.1:
                log.info("Reading all headers took %.3f seconds.", et-st)
            else:
                log.info("Reading all headers took %.1f seconds.", et-st)

        return df

    def _parse_trace(self, fio, itrc, headers_only=False):
        """Parse next trace."""
        tdb = fio.read(32)
        # Check validity of block
        magic = np.frombuffer(tdb, dtype=f"{self._fp.endian}H", count=1)
        if magic != 17442:
            log.error("Trace descriptor block %d does not start 0x4422.", itrc)
            raise ValueError("Invalid trace descriptor block magic number.")

        tdb_size = np.frombuffer(tdb, dtype=f"{self._fp.endian}H", count=1, offset=2)[0]
        ns = np.frombuffer(tdb, dtype=f"{self._fp.endian}I", count=1, offset=8)[0]

        if ns != self.ns:
            log.warning("Number of samples varies in file %s!", self._fp.file)
            self._fp.fixed = False

        if self._fp.datfmt == 3:
            if ns % 4 != 0:
                log.error("Data format code 3 requires that the no. of samples is divisible by 4!")
                log.error("However, ns=%d on trace %d.", ns, itrc)
                raise RuntimeError("SEG2 file violating data format code 3 specs.")
            else:
                nsfmt3 = int((2.5*ns)/2)
                dtype = np.dtype(f"({nsfmt3},){self._fp.endian}"
                                 f"{tools._SEG2DATAFORMAT[self._fp.datfmt]['type']}")
        else:
            dtype = np.dtype(f"({ns},){self._fp.endian}"
                             f"{tools._SEG2DATAFORMAT[self._fp.datfmt]['type']}")

        header = {}
        free_form_str = fio.read(tdb_size-32)
        header = self._parse_free_format_section(header, free_form_str)
        header["NSAMPLES"] = str(ns)
        if headers_only:
            data = None
        else:
            data = np.fromfile(fio, dtype=dtype, count=1).squeeze()
            if self._fp.datfmt == 3:
                # The following conversion code stems from obspy, Lion Krischer,
                # licensed under LGPL v3 - full credits to the original author
                #
                # Convert one's complement to two's complement by adding one to
                # negative numbers.
                one_to_two = (data < 0)
                # The first two bytes (1 word) of every 10 bytes (5 words) contains
                # a 4-bit exponent for each of the 4 remaining 2-byte (int16) samples.
                exponents = data[0::5].view(self._fp.endian + 'u2')
                result = np.empty(ns, dtype=np.int32)
                # Apply the negative correction, then multiply by correct exponent.
                result[0::4] = ((data[1::5] + one_to_two[1::5]) *
                                2**((exponents & 0x000f) >> 0))
                result[1::4] = ((data[2::5] + one_to_two[2::5]) *
                                2**((exponents & 0x00f0) >> 4))
                result[2::4] = ((data[3::5] + one_to_two[3::5]) *
                                2**((exponents & 0x0f00) >> 8))
                result[3::4] = ((data[4::5] + one_to_two[4::5]) *
                                2**((exponents & 0xf000) >> 12))
                data = result

        return data, header
