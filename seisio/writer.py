"""Abstract writer for seismic files."""

import abc
import logging
import numpy as np
import pandas as pd

from . import _ibm2ieee
from . import seisio
from . import tools

log = logging.getLogger(__name__)


class Writer(seisio.SeisIO, abc.ABC):
    """An abstract Writer class for seismic data I/O."""

    @abc.abstractmethod
    def __init__(self, file, mode):
        """Initialize class Writer."""
        super().__init__(file)

        log.info("Output file: %s", self._fp.file)

        self._fp.mode = mode
        if self._fp.mode not in ["w", "a"]:
            raise ValueError(f"Unknown value '{self._fp.mode}' for argument 'mode'.")

        f_exists = self._fp.file.exists()

        if f_exists and self._fp.mode == "w":
            log.warning("File %s already exists, truncating file.", self._fp.file)
            with open(self._fp.file, "wb") as _:
                pass  # truncate file
        elif not f_exists and self._fp.mode == "w":
            with open(self._fp.file, "wb") as _:
                pass  # create file
        elif f_exists and self._fp.mode == "a":
            log.info("File %s already exists, appending to file.", self._fp.file)
            log.warning("Note: there are no consistency cross-checks performed.")
        elif not f_exists and self._fp.mode == "a":
            with open(self._fp.file, "wb") as _:
                pass  # create file
        # subsequent writes can now always use "append" mode

        self._head_written = False
        self._tail_written = False

    def traces_template(self, nt=0, headers_only=False):
        """
        Return a traces or header structured array initialized to zero.

        Parameters
        ----------
        nt : int, optional (default: 0)
            The total number of entries in the array.
        headers_only : bool (deault: False)
            Whether to return a trace header structure only (True) or also
            include the two-dimensional "data" buffer (False).

        Returns
        -------
        Numpy structured array
            An empty header table.
        """
        if headers_only:
            return np.zeros((nt, ), dtype=self._tr.thdtype).squeeze()
        else:
            return np.zeros((nt, ), dtype=self._tr.trdtype).squeeze()

    def headers_template(self, nt=0, pandas=False):
        """
        Return a header table.

        Parameters
        ----------
        nt : int, optional (default: 0)
            The total number of entries in the table (number of traces).
        pandas : bool, optional (default: False)
            Whether to return a header table as Pandas dataframe (True)
            or as standard Numpy structured array (False).

        Returns
        -------
        Numpy structured array or Pandas dataframe
            An empty header table.
        """
        headers = self.traces_template(nt=nt, headers_only=True)
        if not pandas:
            return headers
        else:
            return pd.DataFrame.from_records(headers)

    def _headers_transfer(self, headers, remap=None, silent=False):
        """
        Transfer all header mnemonics of input table to current header table.

        This function is particularly useful if converting between different
        trace header definition tables where perhaps not all mnemonics are
        identical.

        Parameters
        ----------
        headers : Numpy structured array
            The header table (source table) to be transferred to the current
            table definition and the corresponding mnemonics.
        remap : dict
            A dictionary remapping headers from the source table to the output
            target table.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).

        Returns
        -------
        Numpy structured array
            The new header table (target table).
        """
        if headers is None:
            raise ValueError("No headers given on input.")

        nt = len(headers)
        myheaders = self.headers_template(nt=nt)

        if nt == 0:
            return myheaders

        nn = len(headers.dtype.names)
        transferred = np.ones((nn,), dtype=int)

        # transfer
        for i, mnemonic in enumerate(headers.dtype.names):
            if mnemonic in self.mnemonics:
                myheaders[mnemonic] = headers[mnemonic].copy()
                transferred[i] = 0

        # remap; deliberately handled separately from the transfer
        if remap is not None:
            for i, mnemonic in enumerate(headers.dtype.names):
                newkey = remap.get(mnemonic)
                if newkey is not None:
                    if newkey in self.mnemonics:
                        myheaders[newkey] = headers[mnemonic].copy()
                        if not silent:
                            log.info("Remapping mnemonic '%s' to '%s'.", mnemonic, newkey)
                        transferred[i] = 0
                    elif not silent:
                        log.warning("Cannot remap mnemonic '%s' to '%s', key does not exist in "
                                    "target table.", mnemonic, newkey)

        # check
        idx = np.nonzero(transferred)
        for i in idx[0]:
            log.warning("Mnemonic '%s' dropped (not transferable without remap).",
                        headers.dtype.names[i])

        return myheaders

    def write_traces(self, traces=None, data=None, headers=None, remap=None, silent=False):
        """
        Write seismic traces to disk.

        Either traces or data and headers have to be supplied. The headers can
        be in Numpy structured array format, or in a Pandas dataframe format.

        Trace headers and the data will potentially be remapped, transferred
        to the output header table, converted to the output data type, and
        byteswapped if necessary.

        Parameters
        ----------
        traces : Numpy structured array
            A structured array of seismic traces where the actual data are
            stored as key "data" together with the trace headers. This is the
            format that the seisio reader methods use.
        data : 2D Numpy array
            The pure seismic data as two-dimensional Numpy array. If the data
            are not supplied in the correct output data type, they will be
            converted (attention: potential loss of precision).
        headers : Numpy structured array or Pandas dataframe
            The seismic trace headers corresponding to the data. The length of
            the headers array / dataframe must agree with the length of the
            trace-dimension of the data array.
        remap : dict
            A dictionary remapping headers from the existing table in the
            supplied buffer to the output trace header table to be written to
            disk. The key of the dictionary must exist as header mnemonic in
            the supplied buffer, the value of the key must be a mnemonic
            valid for the selected output trace header table. For example,
            { "cdpt": "tracl" } would remap the header 'cdpt' in the input to
            the 'tracl' header in the outout.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).

        Returns
        -------
        int
            The number of traces written in this call.
        """
        if data is None and headers is None and traces is None:
            raise ValueError("No traces (data, headers) to write to disk.")
        if traces is not None and (data is not None or headers is not None):
            raise ValueError("Need either 'traces' or 'data' plus 'headers'.")
        if (headers is None and data is not None) or (data is None and headers is not None):
            raise ValueError("Need both 'data' and 'headers' to write traces to disk.")

        if not self._head_written:
            raise RuntimeError("File has not been initialized; call 'init' first.")
        if self._tail_written:
            raise RuntimeError("File has already been finalized by 'finalize' function call.")

        if traces is not None:
            nt = len(traces)
            if nt == 0:
                raise ValueError("No traces given, structure is empty.")
            try:
                nt, ns = np.shape(traces["data"])
            except ValueError:
                log.error("Supplied Numpy structure 'traces' has no key called 'data'.")
                raise
            if ns == 0:
                raise ValueError("Traces have zero samples.")
            if ns != self._dp.ns:
                raise ValueError("The traces' number of samples differs from constructor value.")

            keys = list(traces.dtype.names)
            keys.remove("data")
            outheaders = self._headers_transfer(traces[keys], remap=remap, silent=silent)
            data = traces["data"]

        if headers is not None:
            nt_h = len(headers)
            nt, ns = data.shape
            if nt == 0:
                raise ValueError("No traces available, data object is empty.")
            if nt != nt_h:
                raise ValueError("Number of traces in data and headers differ.")
            if ns == 0:
                raise ValueError("Traces have zero samples.")
            if ns != self._dp.ns:
                raise ValueError("The traces' number of samples differs from constructor value.")

            if isinstance(headers, pd.DataFrame):
                outheaders = self._headers_transfer(headers.to_records(index=False),
                                                    remap=remap, silent=silent)
            elif isinstance(headers, np.ndarray):
                outheaders = self._headers_transfer(headers, remap=remap, silent=silent)
            else:
                raise TypeError("Given headers object has unsupported type: %s", type(headers))

        if self._fp.datfmt == 1:
            # IBM float
            outdat = _ibm2ieee.ieee2ibm32(data.astype(np.float32), self._fp.endian)
        else:
            if data.dtype.char != self._fp.dtype.char:
                # cast if necessary
                if not silent:
                    log.info("Casting data from type '%s' to '%s'.",
                             data.dtype.char, self._fp.dtype.char)
                outdat = data.astype(self._fp.dtype, copy=True)
            else:
                outdat = data

            if tools._need_swap(outdat.dtype, self._fp.endian):
                # swap bytes if necessary
                if not silent:
                    log.info("Swapping bytes of output data.")
                outdat = outdat.byteswap(inplace=False)

        if not silent:
            log.info("Writing %d trace(s) to disk...", nt)
        with open(self._fp.file, "ab") as fio:
            for i in np.arange(nt):
                fio.write(outheaders[i].tobytes())
                fio.write(outdat[i, :].tobytes())
            self._fp.filesize = fio.tell()

        self._dp.nt += nt
        return nt
