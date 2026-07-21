"""Abstract reader for seismic files."""

import abc
import decorator
import inspect
import logging
import mmap
import numpy as np
import pandas as pd
import time

from dataclasses import dataclass
from numba import jit
from numpy.lib import recfunctions as rfn

from . import _ibm2ieee
from . import seisio
from . import tools
from . import __version__

log = logging.getLogger(__name__)


@decorator.decorator
def _addhist(func, *args, **kwargs):
    """
    This is a decorator that stores information about a function call.
    """
    callargs = inspect.getcallargs(func, *args, **kwargs)
    callargs.pop("self")
    info = f"{func.__name__}(%s)"
    my_kwargs = callargs.pop("kwargs", {})
    arguments = []
    for k, v in callargs.items():
         if isinstance(v, str):
             arguments.append(f"{k}='{v}'")
         else:
             arguments.append(f"{k}={v}")
    for k, v in my_kwargs.items():
         if isinstance(v, str):
             arguments.append(f"{k}='{v}'")
         else:
             arguments.append(f"{k}={v}")
    self = args[0]
    self._idx.hist = info % "::".join(arguments)
    return func(*args, **kwargs)


@jit("(int64,int64,int64,int64)", nopython=True)
def _calc_blocks(start, stride, count, block):
    """Calculate parameters for multi block reads."""
    indices = np.empty((count*block,), dtype=np.int64)
    i = 0
    for _ in range(count):
        idx_s = start
        idx_e = idx_s+block
        for idx in range(idx_s, idx_e):
            indices[i] = idx
            i += 1
        start += stride
    return indices


@jit("(int64,int64)", nopython=True)
def _create_batches(nt, batch_size):
    """Calculate parameters for batch reads."""
    for ii in np.arange(0, nt, batch_size, dtype=np.int64):
        if ii+batch_size > nt:
            yield ii, nt-ii
        else:
            yield ii, batch_size


class Reader(seisio.SeisIO, abc.ABC):
    """An abstract Reader class for seismic data I/O."""

    @dataclass
    class _IDX():
        """Index-related parameters and objects."""

        grp_by: list = None
        srt_by: list = None
        gord: int = 1
        sord: int = 1
        head: np.ndarray = None
        keys: np.array = None
        ne: int = 0
        nte: np.array = None
        maxnte: int = 0
        hist: str = None

    @abc.abstractmethod
    def __init__(self, file):
        """Initialize class Reader."""
        super().__init__(file)

        self._idx = self._IDX()

        log.info("Input file: %s", self._fp.file)
        if not self._fp.file.exists():
            raise ValueError(f"File '{self._fp.file}' does not exist.")

    @property
    def vsi(self):
        """
        Get the (vertical) sampling interval.

        Returns
        -------
        int
            Sampling interval, usually in microunits (e.g., microseconds)
        """
        return self._dp.si

    @property
    def delay(self):
        """
        Get the delay (usually delay recording time) of the first trace.

        Returns
        -------
        int
            Delay, usually in milliunits (e.g., milliseconds)
        """
        return self._dp.delay

    @property
    def vaxis(self):
        """
        Get the sampling times or depths.

        Returns
        -------
        Numpy array
            Sampling values of vertical axis, usually in units.
        """
        dt = self._dp.si * 1e-6
        beg = self._dp.delay * 1e-3
        end = beg + (self._dp.ns-1) * dt
        return np.arange(beg, end+dt/2, dt)

    def _alter_dtype(self, mnemonics, trace=False):
        """Alter an existing trace dtype to provide a subset of headers only."""
        if isinstance(mnemonics, str):
            keys = [mnemonics]
        else:
            keys = mnemonics.copy()
        # check mnemonic(s) exist
        if not set(keys).issubset(self.mnemonics):
            raise ValueError("At least one mnemonic in 'mnemonics' is invalid.")

        if trace:
            # add data back in
            keys.append("data")
            itemsize = self._tr.trsize
        else:
            itemsize = self._tr.thsize

        formats = [self._tr.trdtype.fields[mn][0] for mn in keys]
        offsets = [self._tr.trdtype.fields[mn][1] for mn in keys]
        titles = [self._tr.trdtype.fields[name][2] if len(self._tr.trdtype.fields[name]) == 3 else None for name in keys]

        return tools._create_custom_dtype(keys, formats, offsets,
                                          itemsize, titles=titles)

    def read_all_headers(self, mnemonics=None, silent=False):
        """
        Get trace headers for all traces.

        Parameters
        ----------
        mnemonics : list of strings (default: None)
            A list of trace header mnemonics to read. If None, all header
            mnemonics are read according to the trace header definition
            table currently in use.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).

        Returns
        -------
        Numpy structured array
            Trace header table.
        """
        if not silent:
            log.info("Reading trace headers for all (%d) traces from disk...", self._dp.nt)

        if mnemonics is not None:
            dtype = self._alter_dtype(mnemonics)
        else:
            dtype = self._tr.thdtype

        st = time.time()
        with open(self._fp.file, "rb") as fio:
            with mmap.mmap(fio.fileno(), length=0, access=mmap.ACCESS_READ, offset=0) as mm:
                h = np.ndarray(shape=(self._dp.nt, ), dtype=dtype, buffer=mm,
                               strides=(self.trsize, ), order='F', offset=self._fp.skip).copy()
        et = time.time()

        if not silent:
            diff = et-st
            if diff < 0.1:
                log.info("Reading headers for all traces took %.3f seconds.", et-st)
            else:
                log.info("Reading headers for all traces took %.1f seconds.", et-st)

        return h

    def read_headers(self, *trcno, mnemonics=None, silent=False):
        """
        Get trace headers for one or more traces.

        Parameters
        ----------
        *trcno : int(s)
            The trace numbers (zero-based) to read from disk.
        mnemonics : list of strings (default: None)
            A list of trace header mnemonics to read. If None, all header
            mnemonics are read according to the trace header definition
            table currently in use.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).

        Returns
        -------
        Numpy structured array
            Trace header table.
        """
        trcs = tools._check(trcno)
        nt = len(trcs)
        if nt == 0:
            raise ValueError("No trace numbers requested. Need at least one.")

        if not silent:
            log.info("Reading headers for %d specific traces from disk...", nt)

        if mnemonics is not None:
            dtype = self._alter_dtype(mnemonics)
        else:
            dtype = self._tr.thdtype

        h = np.ndarray((nt, ), dtype=dtype)

        with open(self._fp.file, "rb") as fio:
            for i, trc in enumerate(trcs):
                if trc < 0 or trc >= self._dp.nt:
                    raise ValueError(f"Requested trace no. {trc} out of range [0,{self._dp.nt}).")
                fio.seek(self._fp.skip+trc*self.trsize, 0)
                h[i] = np.fromfile(fio, dtype=dtype, count=1, offset=0)

        return h

    def read_batch_of_headers(self, start=0, nheaders=100, mnemonics=None, silent=False):
        """
        Get trace headers for a certain number of traces starting at a specific trace.

        Parameters
        ----------
        start : int, optional (default: 0)
            The trace number (zero-based) to start reading from disk.
        nheaders : int, optional (default: 100)
            The number of subsequent traces to read, including 'start' itself.
        mnemonics : list of strings (default: None)
            A list of trace header mnemonics to read. If None, all header
            mnemonics are read according to the trace header definition
            table currently in use.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).

        Returns
        -------
        Numpy structured array
            Trace header table.
        """
        if nheaders < 1:
            raise ValueError("Parameter 'nheaders' must be greater or equal 1.")
        if start < 0 or start+nheaders-1 >= self._dp.nt:
            raise ValueError(f"Requested batch of headers out of range [0,{self._dp.nt}).")

        if not silent:
            log.info("Reading headers for %d traces from disk starting at trace %d...",
                     nheaders, start)

        if mnemonics is not None:
            dtype = self._alter_dtype(mnemonics)
        else:
            dtype = self._tr.thdtype

        with open(self._fp.file, "rb") as fio:
            with mmap.mmap(fio.fileno(), length=0, access=mmap.ACCESS_READ, offset=0) as mm:
                h = np.ndarray(shape=(nheaders, ), dtype=dtype, buffer=mm,
                               strides=(self.trsize, ),
                               offset=self._fp.skip+start*self.trsize, order='F').copy()

        return h

    def read_multibatch_of_headers(self, start=0, count=None, stride=None,
                                   block=None, mnemonics=None, silent=False):
        """
        Get multiple batches of trace headers from the seismic file.

        For instance, start=1, count=3, stride=4, block=2 would get you the
        following trace headers: 1, 2, 5, 6, 9, 10 - the start is 1, 2 traces
        are within each block, the stride from the first trace of a block to
        the first trace in the next block is 4 and in total 3 blocks are read.

        For a data set with, for instance, a constant number of traces per
        gather (say, 480) and 500 gathers in total, this function allows you
        to read the first 10 trace headers within each gather using start=0,
        block=10, stride=480, and count=500.

        Parameters
        ----------
        start : int, optional (default: 0)
            The trace number (zero-based) at which to start reading from disk.
        count : int
            The total number of blocks to read.
        stride : int
            The stride between the first traces in each block.
        block : int
            The size of each block.
        mnemonics : list of strings (default: None)
            A list of trace header mnemonics to read. If None, all header
            mnemonics are read according to the trace header definition
            table currently in use.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).

        Returns
        -------
        Numpy structured array
            Trace header table.
        """
        if start < 0 or start >= self._dp.nt:
            raise ValueError(f"Requested batch of trace headers out of range [0,{self._dp.nt}).")
        if count is None or stride is None or block is None:
            raise ValueError("Need to specify count, stride and block.")

        if not silent:
            log.info("Reading %d block(s) of %d trace header(s) from disk, "
                     "starting at %d with stride %d...", count, block, start, stride)

        indices = _calc_blocks(start, stride, count, block)

        if np.max(indices) >= self._dp.nt:
            raise ValueError("Requested multibatch of trace headers out of "
                             f"range [0,{self._dp.nt}).")
        nheaders = len(indices)

        if mnemonics is not None:
            dtype = self._alter_dtype(mnemonics)
        else:
            dtype = self._tr.thdtype

        h = np.ndarray((nheaders, ), dtype=dtype)

        with open(self._fp.file, "rb") as fio:
            for i in np.arange(nheaders):
                fio.seek(self._fp.skip+indices[i]*self.trsize, 0)
                h[i] = np.fromfile(fio, dtype=dtype, count=1, offset=0)

        return h

    def read_dataset(self, mnemonics=None, silent=False, history=None):
        """Get all traces - an alias for read_all_traces()."""
        return self.read_all_traces(mnemonics=mnemonics, silent=silent, history=history)

    def read_all_traces(self, mnemonics=None, silent=False, history=None):
        """
        Get all traces (e.g., read the entire file).

        Parameters
        ----------
        mnemonics : list of strings (default: None)
            A list of trace header mnemonics to read. If None, all header
            mnemonics are read according to the trace header definition
            table currently in use.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).
        history : list, optional (default: None)
            Processing history as list of strings.

        Returns
        -------
        Numpy structured array
            Trace headers and data.
        """
        if not silent:
            log.info("Reading entire file (%d traces) from disk...", self._dp.nt)

        if mnemonics is not None:
            dtype = self._alter_dtype(mnemonics, trace=True)
        else:
            dtype = self._tr.trdtype

        st = time.time()
        with open(self._fp.file, "rb") as fio:
            with mmap.mmap(fio.fileno(), length=0, access=mmap.ACCESS_READ, offset=0) as mm:
                d = np.ndarray(shape=(self._dp.nt, ), dtype=dtype, buffer=mm,
                               offset=self._fp.skip, order='F').copy()
        et = time.time()

        if not silent:
            diff = et-st
            if diff < 0.1:
                log.info("Reading all traces took %.3f seconds.", et-st)
            else:
                log.info("Reading all traces took %.1f seconds.", et-st)

        if self._fp.datfmt == 1:
            if not silent:
                log.info("Converting IBM floats to IEEE floats.")
            data = d["data"].view(f"{self._fp.endian}u4")
            st = time.time()
            d["data"] = _ibm2ieee.ibm2ieee32(data, self._fp.endian)
            et = time.time()
            if not silent:
                diff = et-st
                if diff < 0.1:
                    log.info("Converting all traces took %.3f seconds.", et-st)
                else:
                    log.info("Converting all traces took %.1f seconds.", et-st)

        if history is not None:
            history.append(f"seisio {__version__}: read entire data set '{self._fp.file.absolute()}', "
                           f"ntraces={self._dp.nt:d}, nsamples={self._dp.ns:d}.")

        return d

    def read_traces(self, *trcno, mnemonics=None, silent=False, history=None):
        """
        Get one or more traces.

        Parameters
        ----------
        *trcno : int(s)
            The trace numbers (zero-based) to read from disk.
        mnemonics : list of strings (default: None)
            A list of trace header mnemonics to read. If None, all header
            mnemonics are read according to the trace header definition
            table currently in use.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).
        history : list, optional (default: None)
            Processing history as list of strings.

        Returns
        -------
        Numpy structured array
            Trace headers and data
        """
        trcs = tools._check(trcno)
        nt = len(trcs)
        if nt == 0:
            raise ValueError("No trace numbers requested. Need at least one.")

        if not silent:
            log.info("Reading %d specific trace(s) from disk...", nt)

        if mnemonics is not None:
            dtype = self._alter_dtype(mnemonics, trace=True)
        else:
            dtype = self._tr.trdtype

        d = np.ndarray((nt, ), dtype=dtype)

        with open(self._fp.file, "rb") as fio:
            for i, trc in enumerate(trcs):
                if trc < 0 or trc >= self._dp.nt:
                    raise ValueError(f"Requested trace no. {trc} out of range [0,{self._dp.nt}).")
                fio.seek(self._fp.skip+trc*self.trsize, 0)
                d[i] = np.fromfile(fio, dtype=dtype, count=1, offset=0)

        if self._fp.datfmt == 1:
            data = d["data"].view(f"{self._fp.endian}u4")
            d["data"] = _ibm2ieee.ibm2ieee32(data, self._fp.endian)

        if history is not None:
            history.append(f"seisio {__version__}: read traces from '{self._fp.file.absolute()}', "
                           f"trace numbers=[{', '.join(str(x) for x in trcs)}], "
                           f"ntraces={nt:d}, nsamples={self._dp.ns:d}.")

        return d

    def read_batch_of_traces(self, start=0, ntraces=100, mnemonics=None,
                             silent=False, history=None):
        """
        Get a certain number of traces starting at a specific trace.

        Parameters
        ----------
        start : int, optional (default: 0)
            The trace number (zero-based) to start reading from disk.
        ntraces : int, optional (default: 100)
            The number of subsequent traces to read, including 'start' itself.
        mnemonics : list of strings (default: None)
            A list of trace header mnemonics to read. If None, all header
            mnemonics are read according to the trace header definition
            table currently in use.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).
        history : list, optional (default: None)
            Processing history as list of strings.

        Returns
        -------
        Numpy structured array
            Trace headers and data.
        """
        if ntraces < 1:
            raise ValueError("Parameter 'ntraces' must be greater or equal 1.")
        if start < 0 or start+ntraces-1 >= self._dp.nt:
            raise ValueError(f"Requested batch of traces out of range [0,{self._dp.nt}).")

        if not silent:
            log.info("Reading %d trace(s) from disk starting at trace %d...", ntraces, start)

        if mnemonics is not None:
            dtype = self._alter_dtype(mnemonics, trace=True)
        else:
            dtype = self._tr.trdtype

        with open(self._fp.file, "rb") as fio:
            fio.seek(self._fp.skip+start*self.trsize, 0)
            d = np.fromfile(fio, dtype=dtype, count=ntraces, offset=0)

        if self._fp.datfmt == 1:
            data = d["data"].view(f"{self._fp.endian}u4")
            d["data"] = _ibm2ieee.ibm2ieee32(data, self._fp.endian)

        if history is not None:
            history.append(f"seisio {__version__}: read traces from '{self._fp.file.absolute()}', "
                           f"first trace={start:d}, ntraces={ntraces:d}, "
                           f"nsamples={self._dp.ns:d}.")

        return d

    def read_multibatch_of_traces(self, start=0, count=None, stride=None, block=None,
                                  mnemonics=None, silent=False, history=None):
        """
        Get multiple batches of traces from the seismic file.

        See method read_multibatch_of_headers() for some examples.

        Parameters
        ----------
        start : int, optional (default: 0)
            The trace number (zero-based) at which to start reading from disk.
        count : int
            The total number of blocks to read.
        stride : int
            The stride between the first traces in each block.
        block : int
            The size of each block.
        mnemonics : list of strings (default: None)
            A list of trace header mnemonics to read. If None, all header
            mnemonics are read according to the trace header definition
            table currently in use.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).
        history : list, optional (default: None)
            Processing history as list of strings.

        Returns
        -------
        Numpy structured array
            Trace headers and data.
        """
        if start < 0 or start >= self._dp.nt:
            raise ValueError(f"Requested multibatch of traces out of range [0,{self._dp.nt}).")
        if count is None or stride is None or block is None:
            raise ValueError("Need to specify count, stride and block.")

        if not silent:
            log.info("Reading %d block(s) of %d trace(s) from disk, "
                     "starting at %d with stride %d...", count, block, start, stride)

        indices = _calc_blocks(start, stride, count, block)
        if np.max(indices) >= self._dp.nt:
            raise ValueError(f"Requested multibatch of traces out of range [0,{self._dp.nt}).")
        ntraces = len(indices)

        if mnemonics is not None:
            dtype = self._alter_dtype(mnemonics, trace=True)
        else:
            dtype = self._tr.trdtype

        d = np.zeros((ntraces, ), dtype=dtype)

        with open(self._fp.file, "rb") as fio:
            for i in np.arange(ntraces):
                fio.seek(self._fp.skip+indices[i]*self.trsize, 0)
                d[i] = np.fromfile(fio, dtype=dtype, count=1, offset=0)

        if self._fp.datfmt == 1:
            data = d["data"].view(f"{self._fp.endian}u4")
            d["data"] = _ibm2ieee.ibm2ieee32(data, self._fp.endian)

        if history is not None:
            history.append(f"seisio {__version__}: read traces from '{self._fp.file.absolute()}', "
                           f"first trace={start:d}, block size={block:d}, "
                           f"number of blocks={count:d}, stride={stride:d}, "
                           f"ntraces={ntraces:d}, nsamples={self._dp.ns:d}.")

        return d

    def batches_of_headers(self, batch_size=100, mnemonics=None, silent=False):
        """
        Loop through all headers in blocks (using a generator).

        Parameters
        ----------
        batch_size : int, optional (default: 100)
            The batch size, i.e., number of trace headers to read in one go.
        mnemonics : list of strings (default: None)
            A list of trace header mnemonics to read. If None, all header
            mnemonics are read according to the trace header definition
            table currently in use.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).

        Yields
        ------
        Numpy structured array
            Trace headers.
        """
        nt = np.int64(self._dp.nt)
        if batch_size <= 0:
            raise ValueError("Parameter 'batch_size' cannot be zero or negative.")
        bs = np.int64(batch_size)
        for start, ntraces in _create_batches(nt, bs):
            yield self.read_batch_of_headers(start=start, nheaders=ntraces,
                                             mnemonics=mnemonics, silent=silent)

    def batches(self, batch_size=100, mnemonics=None, silent=False, history=None):
        """
        Loop through all traces in blocks (using a generator).

        Parameters
        ----------
        batch_size : int, optional (default: 100)
            The batch size, i.e., number of traces to read in one go.
        mnemonics : list of strings (default: None)
            A list of trace header mnemonics to read. If None, all header
            mnemonics are read according to the trace header definition
            table currently in use.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).
        history : list, optional (default: None)
            Processing history as list of strings.

        Yields
        ------
        Numpy structured array
            Trace headers and data.
        """
        nt = np.int64(self._dp.nt)
        if batch_size <= 0:
            raise ValueError("Parameter 'batch_size' cannot be zero or negative.")
        bs = np.int64(batch_size)
        for start, ntraces in _create_batches(nt, bs):
            yield self.read_batch_of_traces(start=start, ntraces=ntraces,
                                            mnemonics=mnemonics, silent=silent,
                                            history=history)

    def traces(self, mnemonics=None, silent=False, history=None):
        """
        Loop through all traces of the file (using a generator).

        Parameters
        ----------
        mnemonics : list of strings (default: None)
            A list of trace header mnemonics to read. If None, all header
            mnemonics are read according to the trace header definition
            table currently in use.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).
        history : list, optional (default: None)
            Processing history as list of strings.

        Yields
        ------
        Numpy structured array
            Trace headers and data.
        """
        counter = 0
        while counter < self._dp.nt:
            yield self.read_traces(counter, mnemonics=mnemonics,
                                   silent=silent, history=history)
            counter += 1

    def headers(self, mnemonics=None, silent=False):
        """
        Loop through all headers of the file (using a generator).

        Parameters
        ----------
        mnemonics : list of strings (default: None)
            A list of trace header mnemonics to read. If None, all header
            mnemonics are read according to the trace header definition
            table currently in use.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).

        Yields
        ------
        Numpy structured array
            Trace headers.
        """
        hcounter = 0
        while hcounter < self._dp.nt:
            yield self.read_headers(hcounter, mnemonics=mnemonics, silent=silent)
            hcounter += 1

    def read_vslice(self, n=None, reshape=True, idef="xline", jdef="iline",
                    is_sorted=False, header_trid="trid", fill_value=np.nan,
                    silent=False, history=None):
        """
        Get a time or depth slice.

        For additional details on how the data read from disk might get
        reshaped or padded, please check the ensemble2cube function.

        Parameters
        ----------
        n : int, optional (default: None)
            The vertical slice number to read. Value must be in range [0,ns).
            If None, then ns//2 is chosen as default.
        reshape : bool, optional (default: True)
            Whether to reshape the data read from disk into a 2D time or depth
            slice via function ensemble2cube.
        idef : str, optional (default: 'xline')
            The header mnemonic present in the ensemble's trace headers that
            remains constant along the i-axis.
        jdef : str, optional (default: 'iline')
            The header mnemonic present in the ensemble's trace headers that
            remains constant along the j-axis.
        is_sorted : bool, optional (default: False)
            If the ensemble is already sorted by order=[idef, jdef], set this
            parameter to True to avoid an additional sort (copy). There is no
            check performed whether the ensemble is sorted correctly.
        header_trid : str, optional (default: 'trid')
            Trace header mnemonic to use in order to flag padded traces.
            If set to None, padded traces won't be flagged, otherwise the trace
            identification is set to 3 ('dummy').
        fill_value : numeric value, optional (default: np.nan)
            Fill value for trace positions that get padded.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).
        history : list, optional (default: None)
            Processing history as list of strings.

        Returns
        -------
        Numpy structured array
            Trace headers and tiome/depth slice data
        """
        if n is None:
            n = self._dp.ns//2
        if n < 0 or n >= self._dp.ns:
            raise ValueError(f"Requested vertical slice {n} out of range [0,{self._dp.ns}).")

        if not silent:
            log.info("Reading vertical slice %d from disk...", n)

        # construct cutom dtype
        keys = self._tr.trdtype.names
        formats = [self._tr.trdtype.fields[k][0] for k in keys]
        # read only one sample per trace
        formats[-1] = np.dtype((formats[-1].base, (1,)))
        titles = [self._tr.trdtype.fields[k][2] if len(self._tr.trdtype.fields[k]) == 3 else None for k in keys]
        offsets = [self._tr.trdtype.fields[k][1] for k in keys]
        # offset of sample to read from each trace
        offsets[-1] += n*self._fp.dtype.itemsize
        # itemsize is full trace length
        itemsize = self._tr.trsize
        dtype = tools._create_custom_dtype(keys, formats, offsets, itemsize, titles=titles)

        st = time.time()
        with open(self._fp.file, "rb") as fio:
            with mmap.mmap(fio.fileno(), length=0, access=mmap.ACCESS_READ, offset=0) as mm:
                d = np.ndarray(shape=(self._dp.nt, ), dtype=dtype, buffer=mm,
                               offset=self._fp.skip, order='F').copy()
        et = time.time()

        if not silent:
            diff = et-st
            if diff < 0.1:
                log.info("Reading vertical slice took %.3f seconds.", et-st)
            else:
                log.info("Reading vertical slice took %.1f seconds.", et-st)

        if self._fp.datfmt == 1:
            if not silent:
                log.info("Converting IBM floats to IEEE floats.")
            data = d["data"].view(f"{self._fp.endian}u4")
            st = time.time()
            d["data"] = _ibm2ieee.ibm2ieee32(data, self._fp.endian)
            et = time.time()
            if not silent:
                diff = et-st
                if diff < 0.1:
                    log.info("Converting vertical slice took %.3f seconds.", et-st)
                else:
                    log.info("Converting vertical slice took %.1f seconds.", et-st)

        if history is not None:
            history.append(f"seisio {__version__}: read vertical slice {n:d} from "
                           f"data set '{self._fp.file.absolute()}', reshape={reshape}, "
                           f"idef='{idef}', jdef='{jdef}'.")

        if not reshape:
            return d
        else:
            return tools.ensemble2cube(d, idef=idef, jdef=jdef, is_sorted=is_sorted,
                                       header_trid=header_trid, fill_value=fill_value,
                                       silent=silent)

    @_addhist
    def create_index(self, group_by=None, sort_by=None, group_order=">",
                     sort_order=">", headers=None, filt=None):
        """
        Create a lookup index for the input file to read ensembles.

        In order to form ensembles, i.e., common-midpoint gathers, common-shot
        gathers, common-receiver gathers, etc., with possibly differing number
        of traces, various traces that are not necessarily stored on disk in
        the correct order have to be read and grouped together.

        This method creates a lookup table (index) where groups of traces are
        formed according to user-supplied trace header mnemonics. Each group
        or ensemble can then be sorted by yet another set of user-supplied
        trace header mnemonics. Before grouping takes place, a filter function
        to restrict traces to be considered can be applied. The order in which
        groups are formed as well as the order in which traces are sorted
        within groups can be specified as either ascending or descending.

        Parameters
        ----------
        group_by : string or iterable of strings
            The header mnemonic(s) to form groups of traces (ensembles). At
            least one mnemonic needs to be supplied, and it must be a valid
            trace header key.
        sort_by : string or iterable of strings, optional
            The header mnemonic(s) by which to sort traces within an ensemble.
            If None, then the traces within an ensemble are returned in the
            order they are stored in the file.
        group_order : char, optional (default: ">")
            Sort order for groups, either ">" for ascending or "<" for
            descending.
        sort_order : char, optional (default: ">")
            Sort order within groups. Either ">" for ascending or "<" for
            descending.
        headers : Numpy structured array or None (default: None)
            The trace header table with values for the entire file. If you
            have previously read headers *for all traces* from disk you can
            supply a complete header array here. If none is available, the
            (relevant) headers are read from the disk file.
        filt:
            Filter function to apply before grouping takes place. The filter
            function can refer to all available trace header mnemonics.

        Examples
        --------
        Simple example:
            create index(headers=myheadertable,
                         group_by='cdp',
                         sort_by='offset')
        This will create ensembles where each ensemble has the 'cdp' trace
        header menmonic in common (i.e., CMP gathers). The CMP gathers will
        be sorted in ascending order. The traces within each ensemble will be
        sorted by the 'offset' trace header mnemonic in ascending order.

        Example using a filter function:
            def filt_func(x): return (x['offset'] <= 3000)
            create_index(group_by=["sx", "sy"],
                         sort_by="offset",
                         sort_order="<",
                         filt=filt_func)
        This will create ensembles where each ensemble has the same shot
        coordinates (basically shot gathers) and traces within each ensemble
        are sorted by offset in descending order. Before the groups are formed,
        all traces with offsets larger than 3000 m are removed through the
        application of the filter function. The resulting ensembles will
        therefore not contain any offsets larger than 3000 m.
        """
        self._idx.grp_by = tools._check(group_by)
        self._idx.srt_by = tools._check(sort_by)
        self._idx.gord = 1 if group_order == ">" else -1
        self._idx.sord = 1 if sort_order == ">" else -1

        # ensure specified mnemonics actually exist
        if not set(self._idx.grp_by).issubset(self.mnemonics):
            raise ValueError("At least one mnemonic in 'group_by' is invalid.")
        if not set(self._idx.srt_by).issubset(self.mnemonics):
            raise ValueError("At least one mnemonic in 'sort_by' is invalid.")

        if headers is None:
            try:
                # read only headers we are going to require
                if filt is not None:
                    req = tools._mnemonics_used(filt)
                else:
                    req = set()
                req_head = [*req, *self._idx.grp_by]
                h = self.read_all_headers(mnemonics=req_head)
            except Exception:
                # try reading all headers if reading selected headers failed
                h = self.read_all_headers()
        else:
            h = headers

        # need to store trace index explicitly as filter function could
        # potentially remove entire entries and the buffer slot would no
        # longer match the trace index
        nt = len(h)
        h = tools.add_mnemonic(h, names="index", data=[np.arange(nt)], dtypes=int)

        if filt is not None:
            log.info("Ensemble lookup index has filter applied.")
            h = h[np.nonzero(filt(h))]

        filt_keys = self._idx.grp_by + ["index"]
        self._idx.head = rfn.repack_fields(h[filt_keys], align=False)
        self._idx.keys = np.sort(np.unique(self._idx.head[self._idx.grp_by]),
                                 order=self._idx.grp_by)[::self._idx.gord]

        log.info("Created lookup index for %s (order '%s').", self._idx.grp_by, group_order)
        log.info("Each ensemble is sorted by %s (order '%s').", self._idx.srt_by, sort_order)
        log.info("Number of ensembles: %d", self.ne)

    @property
    def ensemble_keys(self):
        """
        Get the ensemble keys (identifiers) for the current index.

        Returns
        -------
        Numpy array
            Ensemble keys.
        """
        if self._idx.keys is not None:
            return self._idx.keys
        log.warning("No index available. You need to call create_index() first.")
        return np.array([])

    @property
    def ne(self):
        """
        Get the number of ensembles (groups) for the current index.

        Returns
        -------
        int
            Number of ensembles.
        """
        if self._idx.keys is not None:
            return len(self._idx.keys)
        log.warning("No index available. You need to call create_index() first.")
        return 0

    @property
    def nensembles(self):
        """
        Get the number of ensembles (groups) for the current index.

        Returns
        -------
        int
            Number of ensembles.
        """
        return self.ne

    @property
    def nte(self):
        """
        Get the number of traces per ensemble key in this index.

        Returns
        -------
        Numpy array
            Number of traces in each ensemble.
        """
        if self._idx.keys is not None:
            return np.array([len(np.nonzero(self._idx.head[self._idx.grp_by] == x)[0])
                             for x in self._idx.keys[self._idx.grp_by]])
        log.warning("No index available. You need to call create_index() first.")
        return np.array([0])

    @property
    def maxnte(self):
        """
        Get the maximum number of traces found in all ensembles.

        Returns
        -------
        int
            Size of largest ensemble.
        """
        return np.max(self.nte)

    def _get_eidx(self, key):
        if type(key) is np.void:
            key_cmp = key
        else:
            key_cmp = np.asarray(key, dtype=self._idx.head[self._idx.grp_by].dtype)
        return self._idx.head[self._idx.head[self._idx.grp_by] == key_cmp]["index"]

    def read_ensemble(self, *idx_keys, mnemonics=None, silent=False, history=None):
        """
        Get one or more ensembles (groups of traces) from a seismic file.

        Parameters
        ----------
        *idx_keys : tuple(s)
            The keys used to identify ensembles.
        mnemonics : list of strings (default: None)
            A list of trace header mnemonics to read. If None, all header
            mnemonics are read according to the trace header definition
            table currently in use.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).
        history : list, optional (default: None)
            Processing history as list of strings.

        Returns
        -------
        Numpy structured array
            Trace headers and data.
        """
        if self._idx.head is None:
            raise RuntimeError("No index available. You need to call create_index() first.")

        for i, val in enumerate(idx_keys):
            trc = self._get_eidx(val)
            if i == 0:
                traces_to_read = trc
            else:
                traces_to_read = np.union1d(traces_to_read, trc)

        # ensure header mnemonic for sort is available
        if mnemonics is not None:
            if isinstance(mnemonics, str):
                mnemonics = [mnemonics]
            mnemonics = list(set(mnemonics + self._idx.srt_by))

        if not silent:
            log.info("Reading ensemble(s) '%s'.", idx_keys[0])

        if tools._check_if_contiguous(traces_to_read):
            d = self.read_batch_of_traces(start=traces_to_read[0], ntraces=len(traces_to_read),
                                          mnemonics=mnemonics, silent=silent)
        else:
            d = self.read_traces(*traces_to_read, mnemonics=mnemonics, silent=silent)

        if history is not None:
            history.append(f"seisio {__version__}: read traces from '{self._fp.file.absolute()}', "
                           f"ensembles=[{', '.join(str(x) for x in tools._check(idx_keys))}], "
                           f"ntraces={len(traces_to_read):d}, nsamples={self._dp.ns:d}; "
                           f"{self._idx.hist}.")

        return np.sort(d, order=self._idx.srt_by)[::self._idx.sord]

    def ensembles(self, mnemonics=None, silent=False, history=None):
        """
        Loop through all ensembles.

        Parameters
        ----------
        mnemonics : list of strings (default: None)
            A list of trace header mnemonics to read. If None, all header
            mnemonics are read according to the trace header definition
            table currently in use.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).
        history : list, optional (default: None)
            Processing history as list of strings.

        Yields
        ------
        Numpy structured array
            Trace headers and data.
        """
        if self._idx.keys is not None:
            for e in self._idx.keys:
                yield self.read_ensemble(e, mnemonics=mnemonics,
                                         silent=silent, history=history)
        else:
            raise RuntimeError("No index available. You need to call create_index() first.")

    def thstat(self, headers=None, ntmax=None):
        """
        Determine statistics for each trace header mnemonic.

        Parameters
        ----------
        headers : Numpy structured array, optional (default: None)
            The trace header values. If None, then all trace headers will
            be read from disk. If a structured array contains the data as
            well, they will be dropped before calculating the statistics.
        ntmax : int, optional (default: None)
            Maximum number of traces to take into consideration to build
            statistics. Default is None, i.e., all traces are considered.
        """
        log.info("Calculating trace header statistics.")

        if headers is None:
            if ntmax is None:
                h = self.read_all_headers()
            else:
                h = self.read_batch_of_headers(start=0, nheaders=ntmax)
        else:
            if headers.dtype.names is not None:
                keys = list(headers.dtype.names)
                if "data" in keys:
                    keys.remove("data")
                h = headers[keys]
            else:
                raise ValueError("No structured array with trace headers given.")

        if not h.dtype.isnative:
            h = h.view(h.dtype.newbyteorder()).byteswap()

        summary = pd.DataFrame(h).describe().transpose().loc[:, ['min', 'max', 'mean',
                                                                 'std', '25%', '75%']]

        return summary

    def log_thstat(self, thstat=None, traces=None, zero=False, ntmax=None):
        """
        Print statistics for each trace header mnemonic.

        Parameters
        ----------
        thstat : Pandas dataframe, optional (default: None)
            A Pandas dataframe as produced by the 'thstat' function.
            If None, this routine uses the 'headers' argument or, if
            no headers are provided, calls 'thstat' itself.
        traces : Numpy structured array, optional (default: None)
            The seismic traces (trace headers plus data) or the seismic
            trace headers (as provided by the 'read_all_headers' function).
        zero : bool, optional (default: False)
            Do not print entries that have a value of zero (False) or print
            all min/max entries, independent of values (True).
        ntmax : int, optional (default: None)
            Maximum number of traces to take into consideration to build
            statistics. Default is None, i.e., all traces are considered.
            Only relevant if df=None.
        """
        if thstat is None and traces is None:
            df = self.thstat(headers=None, ntmax=ntmax)
        elif thstat is not None and traces is None:
            df = thstat.copy()
        elif thstat is None and traces is not None:
            keys = list(traces.dtype.names)
            if "data" in keys:
                keys.remove("data")
            df = self.thstat(headers=traces[keys], ntmax=ntmax)
        else:
            log.warning("Both 'thstat' and 'traces' provided. Using 'thstat'.")
            df = thstat.copy()

        if zero is False:
            msg = "Summary of trace header statistics (zeros excluded):"
            mask = np.any([df["min"] != 0, df["max"] != 0], axis=0)
            df = df.loc[mask, :]
        else:
            msg = "Summary of trace header statistics (zeros included):"

        try:
            from tabulate import tabulate
            log.info("%s\n%s", msg, tabulate(df, headers="keys", tablefmt="psql"))
        except ImportError:
            log.info("%s", msg)
            log.info("%s\n%s", "-------- BEGIN --------", df.to_markdown())
            log.info("%s", "--------- END ---------")

        return df
