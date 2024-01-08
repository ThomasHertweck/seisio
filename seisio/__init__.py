"""
I/O of seismic files.

The seisio Python module provides basic input/output operations for seismic
data in typical standard formats such as SEG-Y or SU.

In this module, data and trace headers are handled as Numpy structured arrays,
i.e., the full spectrum of functions coming along with Numpy and related
modules can be used. Simplicity and user-friendliness have priority in this
module and its intended use compared to achieving the best performance or
highest throughput. The classes and methods provided are kept deliberately
simple to get students participating our lectures and exercises going with
Python and seismic data in standard industry formats. The classes are not
meant to offer all functionality required in a commercial production
processing environment. Having said that, the module tries to keep
performance in mind (e.g., using memory-mapped files where possible) and also
offers functionality not readily available in other packages that are
available in the Python ecosystem.

Author & Copyright: Dr. Thomas Hertweck, geophysics@email.de

License: GNU Lesser General Public License, Version 3
         https://www.gnu.org/licenses/lgpl-3.0.html
"""

__version__ = "1.1.0"
__author__ = "Thomas Hertweck"
__copyright__ = "(c) 2024 Thomas Hertweck"
__license__ = "GNU Lesser General Public License, Version 3"

import logging
import numpy as np
import pandas as pd
import shutil

from pathlib import Path

from . import seg2
from . import segy
from . import su
from . import _txtheader

log = logging.getLogger(__name__)

_FILE_SUFFIX = [".SGY", ".SEGY", ".SEG-Y", ".SEG_Y",   # SEG-Y
                ".SEG2", ".DAT", ".SG2",               # SEG2
                ".SU"]                                 # SU


def input(file, *args, **kwargs):
    """
    Open a seismic file for reading.

    Parameters
    ----------
    file : str or pathlib.Path
        The name of the file to read.
    filetype : str
        Force a specific file type if file suffix is unknown.

    All other arguments are passed to the underlying class once
    the type of input file is determined.
    """
    filetype = kwargs.pop("filetype", None)

    fpath = Path(file)
    if not fpath.exists():
        raise ValueError(f"Input file {fpath} does not exist.")

    if filetype is None:
        suffix = fpath.suffix.upper()
    else:
        suffix = str("." + filetype).upper()
        if suffix not in _FILE_SUFFIX:
            raise ValueError(f"Unknown value '{filetype}' for argument 'filetype'.")

    if suffix in [".SEGY", ".SGY", ".SEG-Y"]:
        return segy.Reader(file, **kwargs)
    elif suffix in [".SEG2", ".DAT", ".S2", ".SG2"]:
        return seg2.Reader(file, **kwargs)
    elif suffix in [".SU"]:
        return su.Reader(file, **kwargs)
    else:
        raise RuntimeError("Cannot determine file type from file name.")


def output(file, **kwargs):
    """
    Open a seismic file for writing.

    Parameters
    ----------
    file : str or pathlib.Path
        The name of the file to write.
    filetype : str
        Force a specific file type if file suffix is unknown.

    All other arguments are passed to the underlying class once
    the type of output file is determined.
    """
    filetype = kwargs.pop("filetype", None)

    fpath = Path(file)

    if filetype is None:
        suffix = fpath.suffix.upper()
    else:
        suffix = str("." + filetype).upper()
        if suffix not in _FILE_SUFFIX:
            raise ValueError(f"Unknown value '{filetype}' for parameter 'filetype'.")

    if suffix in [".SEGY", ".SGY", ".SEG-Y"]:
        return segy.Writer(file, **kwargs)
    elif suffix in [".SEG2", ".DAT", ".S2", ".SG2"]:
        raise NotImplementedError("SEG2 output not implemented as it is an acq. format.")
    elif suffix in [".SU"]:
        return su.Writer(file, **kwargs)
    else:
        raise RuntimeError("Cannot determine file type from file name.")


def segy_thdef_template(outfile):
    """
    Get a template to be used as SEG-Y trace header definition table.

    The template can be adjusted for trace header mnemonic names, types,
    byte offsets etc. and then used later on to allow for a custom-made
    trace header definition table.

    Parameters
    ----------
    outfile : str
        Target name (or directory) for the JSON.
    """
    shutil.copy(Path(__file__).parent/"json/segy_traceheader.json", outfile)


def segy_bhdef_template(outfile):
    """
    Get a template to be used as SEG-Y binary header definition table.

    The template can be adjusted for binary header mnemonic names, types,
    byte offsets etc. and then used later on to allow for a custom-made
    binary header definition table.

    Parameters
    ----------
    outfile : str
        Target name (or directory) for JSON.
    """
    shutil.copy(Path(__file__).parent/"json/segy_binaryheader.json", outfile)


def segy_thdef1_template(outfile):
    """
    Get a template to be used as SEG-Y trace header ext. 1 definition table.

    The template can be adjusted for header mnemonic names, types, byte
    offsets etc. and then used later on to allow for a custom-made trace
    header extension 1 definition table.

    Parameters
    ----------
    outfile : str
        Target name (or directory) for the JSON.
    """
    shutil.copy(Path(__file__).parent/"json/segy_traceheader_ext1.json", outfile)


def su_thdef_template(outfile):
    """
    Get a template to be used as SU trace header definition table.

    The template can be adjusted for trace header mnemonic names, types,
    byte offsets etc. and then used later on to allow for a custom-made
    trace header definition table.

    Parameters
    ----------
    outfile : str
        Target name (or directory) for the JSON.
    """
    shutil.copy(Path(__file__).parent/"json/su_traceheader.json", outfile)


def segy_txthead_template(major_version=1, minor_version=0, fill=True):
    """
    Get a template for a SEG-Y textual file header.

    Parameters
    ----------
    major_version : int, optional (default: 1)
        Major SEG-Y revision number.
    minor_version : int, optional (default: 0)
        Minor SEG-Y revision number.
    fill : boolean, optional (default: True)
        Fill the individual strings with Cxx convention (True) or not.

    Returns
    -------
    list
        A list of strings, 40 card images (strings), each of 80 characters.
    """
    return _txtheader.TxtHeader().template(major_version=major_version,
                                           minor_version=minor_version,
                                           fill=fill)


def _thstat(traces):
    """
    Determine statistics for each trace header mnemonic.

    Parameters
    ----------
    traces : Numpy structured array
        The seismic traces.
    """
    keys = list(traces.dtype.names)
    if "data" in keys:
        keys.remove("data")
    summary = pd.DataFrame(traces[keys]).describe().transpose().loc[:, ['min', 'max', 'mean',
                                                                        'std', '25%', '75%']]
    return summary


def log_thstat(traces, zero=False):
    """
    Print statistics for each trace header mnemonic.

    Parameters
    ----------
    traces : Numpy structured array
        The seismic traces (trace headers plus data) or the seismic
        trace headers (as provided by the 'read_all_headers' function).
    zero : bool, optional (default: False)
        Do not print entries that have a value of zero (False) or print
        all min/max entries, independent of values (True).
    """
    df = _thstat(traces)

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
