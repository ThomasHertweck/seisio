"""Tools to handle seismic trace headers as Numpy structured arrays."""

import logging
import numpy as np

from collections.abc import Iterable
from numpy.lib import recfunctions as rfn
from sys import byteorder

log = logging.getLogger(__name__)

# (SEG-Y) data formats
_DATAFORMAT = {1: {"desc": "4-byte IBM floating-point", "type": "f", "dtype": np.float32},
               2: {"desc": "4-byte two's complement integer", "type": "i", "dtype": np.int32},
               3: {"desc": "2-byte two's complement integer", "type": "h", "dtype": np.int16},
               5: {"desc": "4-byte IEEE floating-point", "type": "f", "dtype": np.float32},
               6: {"desc": "8-byte IEEE floating-point", "type": "d", "dtype": np.float64},
               8: {"desc": "1-byte two's complement integer", "type": "b", "dtype": np.int8},
               9: {"desc": "8-byte two's complement integer", "type": "q", "dtype": np.int64},
               10: {"desc": "4-byte unsigned integer", "type": "I", "dtype": np.uint32},
               11: {"desc": "2-byte unsigned integer", "type": "H", "dtype": np.uint16},
               12: {"desc": "8-byte unsigned integer", "type": "Q", "dtype": np.uint64},
               16: {"desc": "1-byte unsigned integer", "type": "B", "dtype": np.uint8}}

# encodings of different data types
_DATAENCODING = {"b": {"dtype": "int8", "size": 1},
                 "B": {"dtype": "uint8", "size": 1},
                 "h": {"dtype": "int16", "size": 2},
                 "H": {"dtype": "uint16", "size": 2},
                 "i": {"dtype": "int32", "size": 4},
                 "I": {"dtype": "uint32", "size": 4},
                 "q": {"dtype": "int64", "size": 8},
                 "Q": {"dtype": "uint64", "size": 8},
                 "f": {"dtype": "float32", "size": 4},
                 "d": {"dtype": "float64", "size": 8}}

_SEG2DATAFORMAT = {1: {"desc": "2-byte two's complement integer", "type": "h", "dtype": np.int16},
                   2: {"desc": "4-byte two's complement integer", "type": "i", "dtype": np.int32},
                   3: {"desc": "20-bit floating-point (SEG-D)", "type": "i2", "dtype": np.int32},
                   4: {"desc": "4-byte IEEE floating-point", "type": "f", "dtype": np.float32},
                   5: {"desc": "8-byte IEEE floating-point", "type": "d", "dtype": np.float64}}
# untested: format 3 = 20-bit floating point (SEG-D), stored as scaled 32-bit integer

# Note: the order for SEG2 file header entries is important
_SEG2FILEDESCSTR = ["ACQUISITION_DATE", "ACQUISITION_TIME", "CLIENT", "COMPANY",
                    "GENERAL_CONSTANT", "INSTRUMENT", "JOB_ID", "OBSERVER",
                    "PROCESSING_DATE", "PROCESSING_TIME", "TRACE_SORT", "UNITS", "NOTE"]

# Note: the order for SEG2 trace header entries is important
_SEG2TRACEDESCSTR = ["ALIAS_FILTER", "AMPLITUDE_RECOVERY", "BAND_REJECT_FILTER",
                     "CDP_NUMBER", "CDP_TRACE", "CHANNEL_NUMBER", "DATUM", "DELAY",
                     "DESCALING_FACTOR", "DIGITAL_BAND_REJECT_FILTER",
                     "DIGITAL_HIGH_CUT_FILTER", "DIGITAL_LOW_CUT_FILTER",
                     "END_OF_GROUP", "FIXED_GAIN", "HIGH_CUT_FILTER", "LINE_ID",
                     "LOW_CUT_FILTER", "NOTCH_FREQUENCY", "POLARITY", "RAW_RECORD",
                     "RECEIVER", "RECEIVER_GEOMETRY", "RECEIVER_LOCATION",
                     "RECEIVER_SPECS", "RECEIVER_STATION_NUMBER", "SAMPLE_INTERVAL",
                     "SHOT_SEQUENCE_NUMBER", "SKEW", "SOURCE", "SOURCE_GEOMETRY",
                     "SOURCE_LOCATION", "SOURCE_STATION_NUMBER", "STACK",
                     "STATIC_CORRECTIONS", "TRACE_TYPE", "NOTE"]

_SEG2TRACEDESCALIAS = ["Anti-aliasing filter specs", "Amplitude recovery method",
                       "Acquisition band-rejection filter specs", "CDP number",
                       "Trace number within CDP", "Channel number", "Datum (elevation)",
                       "Delay recording time", "Descaling factor",
                       "Processing digital band-rejection filter specs",
                       "Processing digital high-cut filter specs",
                       "Processing digital low-cut filter specs", "Last trace of group flag",
                       "Recording instrument fixed gain (dB)", "Acquisition high-cut filter specs",
                       "Line identification", "Acquisition low-cut filter specs",
                       "Notch filter frequency", "Polarity", "File name of raw record",
                       "Type of receiver (and number of rec. in group)", "Receiver group geometry",
                       "Receiver (group) location (x, y, z)", "Receiver specs",
                       "Receiver station number", "Sampling interval (s)", "Shot sequence number",
                       "Skew value", "Type of source", "Source (array) geometry",
                       "Source (array) location (x, y, z)", "Source station number",
                       "Stack (no. of summed shots)", "Static correction (src, rec, total)",
                       "Trace type", "Further comments"]


def _check(para):
    """Ensure a list exists."""
    if para is not None:
        return list(np.atleast_1d(para))
    return []


def _check_if_contiguous(buffer):
    """Check if a buffer contains contiguous numbers with stride 1."""
    if len(buffer) < 2:
        return 0
    grad = np.diff(buffer)
    if grad.min() == 1 and grad.max() == 1:
        return 1
    else:
        return 0


def _foreign_endian():
    """Return foreign endianess."""
    if byteorder == "little":
        return ">"
    else:
        return "<"


def _native_endian():
    """Return native endianess."""
    if byteorder == "little":
        return "<"
    else:
        return ">"


def _need_swap(dtype, endian="<"):
    """Check whether byte-swapping is required, dependent on Numpy dtype."""
    if endian == "=":
        endian = _native_endian()

    if dtype.isnative:
        if byteorder == "little":
            retval = False if endian == "<" else True
        else:
            retval = False if endian == ">" else True
    else:
        if byteorder == "little":
            retval = True if endian == "<" else False
        else:
            retval = True if endian == ">" else False
    return retval


def _parse_hdef(hdict, endian="="):
    """Parse JSON header definition."""
    hkeys = list(hdict.keys())
    hformats = []
    bytepos = 1

    for key in hkeys:
        keytype = hdict[key]["type"]
        keysize = _DATAENCODING[keytype]["size"]
        hformats.append(f"{endian}{keytype}")
        if bytepos != hdict[key]["byte"]:
            raise ValueError(f"JSON has gaps or overlaps in byte positions at mnemonic {key}, "
                             f"bytepos {bytepos}, expected {hdict[key]['byte']}.")
        bytepos += keysize

    htitles = [hdict[k]["desc"] for k in hkeys]

    return hkeys, hformats, htitles


def _create_dtype(names, formats, titles=None):
    """Create Numpy dtype."""
    if titles is None:
        dtype = np.dtype({"names": names, "formats": formats}, align=False)
    else:
        dtype = np.dtype({"names": names, "formats": formats, "titles": titles}, align=False)

    return dtype


def add_mnemonic(headers, names=None, data=None, dtypes=None):
    """
    Add mnemonic(s) to structured array.

    This function can be used to add, for instance, a trace header
    mnemonic to the corresponding Numpy structured array.

    Parameters
    ----------
    headers : Numpy structured array
        The header structure (e.g., trace headers).
    names : str or list of str
        The trace header mnemonics to add.
    data : value or list of values, array or list of arrays, None
        The data with which to fill the new header slots. If None,
        then the entries will be filled with zeros. If a single value
        is given, or a list of values where the length of the list
        corresponds to the number of mnemonics to add, each value will
        be used to initialize the corresponding new header slots. If
        an array is given where the length of the array corresponds to
        the number of traces in the header array, or a list of arrays
        where each individual array's length equals the number of
        traces, then each array will be used to initialize the
        corresponding new header slots.
    dtypes : data type or list of data types
        The data types for the new header mnemonics. If only a single
        data type is given but multiple mnemonics are added, then the
        single data type will be used for all new mnemonics.

    Returns
    -------
    Numpy structured array
        The original header array with the new mnemonics added and
        initialized.
    """
    if names is None:
        raise ValueError("Need at least one mnemonic name to add.")
    if dtypes is None:
        raise ValueError("Need to specify dtypes for the new header mnemonic(s).")

    nt = len(headers)

    if isinstance(names, (tuple, list)):
        keys = names
    else:
        keys = [names, ]
    nk = len(keys)

    dt = np.atleast_1d(dtypes)
    ndt = len(dt)
    if nk != ndt:
        if ndt == 1:
            for i in np.arange(nk-ndt):
                dt = np.append(dt, dt[0])
        else:
            raise ValueError("Parameter dtypes must be a single dtype or a list "
                             "of dtypes that matches the number of new mnemonics.")

    val = np.atleast_1d(data)
    nv = len(val)
    if val is None:
        newv = [np.zeros(nt) for i in np.arange(nk)]
    elif nk != nv:
        if nv == 1:
            if isinstance(val[0], Iterable):
                if len(val[0]) != nt:
                    raise ValueError("Length of data does not match number of traces.")
                newv = [val[0] for i in np.arange(nk)]
            else:
                newv = [val[0]*np.ones(nt) for i in np.arange(nk)]
        else:
            raise ValueError("Number of data entries does not match number of new mnemonics.")
    else:
        newv = []
        for v in val:
            if isinstance(v, Iterable):
                if len(v) != nt:
                    raise ValueError("Length of data does not match number of traces.")
                newv.append(v)
            else:
                newv.append(v*np.ones(nt))

    if not headers.dtype.isnative:
        headers = headers.newbyteorder().byteswap()

    return rfn.append_fields(headers, keys, data=newv, dtypes=dt.tolist(), usemask=False)


def remove_mnemonic(headers, names=None, allzero=False):
    """
    Remove mnemonic(s) from structured array.

    This function can be used to remove, for instance, a trace header
    mnemonic from the corresponding Numpy structured array. If you remove
    the "data" mnemonic, you end up with just the trace headers but no
    data values anymore.

    Parameters
    ----------
    headers : Numpy structured array
        The header structure (e.g., trace headers).
    names : str or list of str (default: None)
        The trace header mnemonics to remove.
    allzero : bool (default: False)
        If True, all mnemonics that contain only zeros will be removed,
        possibly in addition to the mnemonics specified via 'names'.

    Returns
    -------
    Numpy structured array
        The original header array with the specified mnemonics removed.
    """
    remove = set()

    if names is None and not allzero:
        raise ValueError("Need at least one mnemonic to remove, or allzero=True.")

    if isinstance(names, str):
        remove.add(names)
    elif isinstance(names, list):
        remove.update(set(names))
    elif isinstance(names, set):
        remove.update(names)

    if allzero:
        keys = list(headers.dtype.names)
        keys.remove("data")
        for k in keys:
            if headers[k].min() == 0 and headers[k].max() == 0:
                remove.add(k)

    return rfn.drop_fields(headers, remove, usemask=False, asrecarray=False)


def rename_mnemonic(headers, mapping=None):
    """
    Rename mnemonic(s) in structured array.

    This function can be used to rename, for instance, a trace header
    mnemonic in the corresponding Numpy structured array.

    Parameters
    ----------
    headers : Numpy structured array
        The header structure (e.g., trace headers).
    namemap : dict
        Dictionary mapping old name(s) to new name(s).

    Returns
    -------
    Numpy structured array
        The original header array with the specified mnemonics renamed.
    """
    if mapping is None:
        raise ValueError("Need a dictionary to map old names to new names.")

    return rfn.rename_fields(headers, mapping)
