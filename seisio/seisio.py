"""SeisIO abstract base class."""

import abc
import json
import logging
import numpy as np
import pandas as pd
import pathlib

from dataclasses import dataclass, field

from . import tools
from . import _txtheader

log = logging.getLogger(__name__)


class SeisIO(abc.ABC):
    """Abstract base class for I/O of seismic files."""

    @dataclass
    class _FP():
        """File-related parameters."""

        file: pathlib.Path = None
        filesize: int = 0
        endian: str = None
        mode: str = None
        fixed: bool = True
        skip: int = 0
        datfmt: int = None
        dtype: np.dtype = None

    @dataclass
    class _DP():
        """Data-related parameters."""

        nt: int = 0
        ns: int = 0
        si: int = 0
        delay: int = 0

    @dataclass
    class _TR():
        """Trace-related parameters and objects."""

        thdict: dict = field(default_factory=dict)
        thsize: int = 0
        trsize: int = 0
        thdtype: np.dtype = None
        trdtype: np.dtype = None

    @dataclass
    class _SEGY():
        """SEG-Y-related parameters and objects."""

        major: int = 0
        minor: int = 0
        bhdict: dict = None
        bhdtype: np.dtype = None
        txthead: _txtheader.TxtHeader = None
        binhead: np.ndarray = None
        txtrec: list = field(default_factory=list)
        txtrail: list = field(default_factory=list)

    @dataclass
    class _SEG2():
        """SEG2-related parameters and objects."""

        sterm: str = None
        lterm: str = None
        trcptr: np.ndarray = None
        fheader: dict = field(default_factory=dict)
        theader: dict = field(default_factory=dict)

    @abc.abstractmethod
    def __init__(self, file):
        """Initialize class SeisIO."""
        self._fp = self._FP()
        self._dp = self._DP()
        self._tr = self._TR()
        self._sgy = None
        self._sg2 = None
        self._par = {}
        self._fp.file = pathlib.Path(file)

    @property
    def endianess(self):
        """
        Get endianess of file.

        Returns
        -------
        char
            "<" (little endian), ">" (big endian).
        """
        return self._fp.endian

    @property
    def file(self):
        """
        Get name of file.

        Returns
        -------
        pathlib.Path
            Name of file.
        """
        return self._fp.file

    @property
    def fsize(self):
        """
        Get the file size.

        Returns
        -------
        int
            Size of file in bytes
        """
        return self._fp.filesize

    @property
    def thsize(self):
        """
        Get the size of a trace header.

        Returns
        -------
        int
            Size of one trace header in bytes.
        """
        return self._tr.thsize

    @property
    def trsize(self):
        """
        Get the size of one complete trace (trace header plus data).

        Returns
        -------
        int
            Size of one complete trace in bytes.
        """
        return self._tr.trsize

    @property
    def ns(self):
        """
        Get the number of samples per trace.

        Returns
        -------
        int
            Number of samples.
        """
        return self._dp.ns

    @property
    def nt(self):
        """
        Get the total (current) number of traces in file.

        Returns
        -------
        int
            Number of traces.
        """
        return self._dp.nt

    @property
    def mnemonics(self):
        """
        Get the trace header mnemonics (keys).

        Returns
        -------
        list : All trace header mnemonics (keys) as list.
        """
        return list(self._tr.thdict.keys())

    def log_thdef(self):
        """Log the used trace header definition."""
        msg = "Trace header definition:"

        df = pd.DataFrame(self._tr.thdict).T
        df["byte"] = df["byte"]-1
        df.rename(columns={"desc": "description"}, inplace=True)
        if self._sg2 is not None:
            df.rename(columns={"byte": "no."}, inplace=True)

        try:
            from tabulate import tabulate
            log.info("%s\n%s", msg, tabulate(df, headers="keys", tablefmt="psql"))
        except ImportError:
            log.info("%s", msg)
            log.info("%s\n%s", "-------- BEGIN --------", df.to_markdown())
            log.info("%s", "--------- END ---------")

    def _log_bhdef(self):
        """Log the SEG-Y binary file header definition."""
        msg = "SEG-Y binary file header definition:"
        if self._sgy is None or self._sgy.bhdict is None:
            log.warning("Cannot log binary file header definition, no header available.")
            return

        df = pd.DataFrame(self._sgy.bhdict).T
        df["byte"] = df["byte"]-1
        df.rename(columns={"desc": "description"}, inplace=True)

        try:
            from tabulate import tabulate
            log.info("%s\n%s", msg, tabulate(df, headers="keys", tablefmt="psql"))
        except ImportError:
            log.info("%s", msg)
            log.info("%s\n%s", "-------- BEGIN --------", df.to_markdown())
            log.info("%s", "--------- END ---------")

    def _log_binhead(self, binhead=None, zero=False):
        """
        Log the SEG-Y binary file header.

        Parameters
        ----------
        binhead : Numpy structured array, optional (default: None)
            The binary header. If 'None', the internally stored binary header
            (if available) is used.
        zero : boolean, optional (default: False)
            Whether to print binary header entries that are zero or not.
        """
        msg = "SEG-Y binary file header:"

        if binhead is None:
            if self._sgy is not None and self._sgy.binhead is not None:
                binhead = self._sgy.binhead
            else:
                log.warning("Cannot log binary file header, no header available.")
                return

        keys = binhead.dtype.names
        bdict = {}
        for k in keys:
            tpl = binhead.dtype.fields[k]
            bdict[k] = {'value': binhead[k][0], 'description': tpl[2]}

        df = pd.DataFrame(bdict).T
        if not zero:
            df = df[df["value"] != 0]

        try:
            from tabulate import tabulate
            log.info("%s\n%s", msg, tabulate(df, headers="keys", tablefmt="psql"))
        except ImportError:
            log.info("%s", msg)
            log.info("%s\n%s", "-------- BEGIN --------", df.to_markdown())
            log.info("%s", "--------- END ---------")

        return binhead

    def _log_txthead(self, txthead=None, info=None):
        """
        Log the (primary) textual file header.

        txthead : list of strings or string, optional (default: None)
            The textual header. If 'None', the internally stored textual
            header (if available) is used.
        info : str, optional (default: None)
            A verbal description of this textual header. Only used if an
            external textual header is provided.
        """
        if txthead is not None:
            if info is None:
                info = "User-supplied textual header"
            txh = _txtheader.TxtHeader(encoding="ebcdic", info=info)
            txh.set_header(txthead, silent=True)
            txh.log_txthead()
            return txthead
        else:
            if self._sgy is None:
                log.warning("Cannot log textual file header, no header available.")
                return
            self._sgy.txthead.log_txthead()
            return self._sgy.txthead.header

    def _set_segy_dtypes(self):
        """Set various dtypes."""
        self._fp.dtype = np.dtype(f"{self._fp.endian}{tools._DATAFORMAT[self._fp.datfmt]['type']}")

        with open(self._par["thdef"], "r") as io:
            thdict = json.load(io)

        th_k, th_f, th_t = tools._parse_hdef(thdict, endian=self._fp.endian)

        if self._par["thext1"]:
            if self._par["thdef1"] is None:
                self._par["thdef1"] = pathlib.Path(__file__).parent/"json/segy_traceheader_ext1.json"
            with open(self._par["thdef1"], "r") as io:
                th1dict = json.load(io)
            th1_k, th1_f, th1_t = tools._parse_hdef(th1dict)
            th_k += th1_k
            th_f += th1_f
            th_t += th1_t

        if self._par["nthuser"] > 0:
            if self._par["thdefu"] is None:
                raise ValueError("User-defined trace headers present but no 'thdefu' "
                                 "argument given.")
            thud = tools._check(self._par["thdefu"])
            if len(thud) != self._par["nthuser"]:
                raise ValueError(f"Number of user-defined trace headers ({self._par['nthuser']}) "
                                 f"and provided number of JSONs ({len(thud)}) do not match.")
            for hd in thud:
                with open(hd, "r") as io:
                    hdict = json.load(io)
                k, f, t = tools._parse_hdef(hdict)
                th_k += k
                th_f += f
                th_t += t
        else:
            if self._par["thdefu"] is not None:
                log.warning("Argument 'thdefu' ignored as no user-defined headers present.")

        self._tr.thdtype = tools._create_dtype(th_k, th_f, th_t)
        self._tr.thsize = self._tr.thdtype.itemsize

        for k in self._tr.thdtype.names:
            tpl = self._tr.thdtype.fields[k]
            t = tpl[0].char
            b = tpl[1]+1
            d = tpl[2]
            self._tr.thdict[k] = {'byte': b, 'type': t, 'desc': d}

        th_k.append("data")
        th_f.append(f"({self._dp.ns},){self._fp.endian}"
                    f"{tools._DATAFORMAT[self._fp.datfmt]['type']}")
        th_t.append("Seismic data")

        self._tr.trdtype = tools._create_dtype(th_k, th_f, th_t)
        self._tr.trsize = self._tr.trdtype.itemsize
