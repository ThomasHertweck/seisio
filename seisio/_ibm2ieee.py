"""Conversion of IBM floats to IEEE floats and vice versa."""

import numpy as np
from numba import jit, vectorize


def ibm2ieee32(ibm, endian):
    """
    Convert IBM floating point numbers to IEEE format.

    Parameters
    ----------
    ibm : np.uint32
        The IBM float(s) (as uint32) to convert.
    endian : char
        Output endianess: ">" for big endian, "<" for little endian.

    Returns
    -------
    np.float32
        IEEE float or array of IEEE floats.
    """
    return _numba_ibm2ieee32_vector(ibm.astype("<u4")).astype(f"{endian}f")


@jit("float32(uint32)", nopython=True, cache=True)
def _numba_ibm2ieee32_single(ibm):
    ieee_sign = ibm & 0x80000000
    ibm_frac = int(ibm & 0x00ffffff)
    if not ibm_frac:
        return np.int32(ieee_sign).view(np.float32)
    ibm_expt = int((ibm & 0x7f000000) >> 22)
    top_digit = ibm_frac & 0x00f00000
    while top_digit == 0:
        ibm_frac <<= 4
        ibm_expt -= 4
        top_digit = ibm_frac & 0x00f00000
    leading_zeros = (int)((0x000055af >> (top_digit >> 19)) & 3)
    ibm_frac <<= leading_zeros
    ieee_expt = ibm_expt - 131 - leading_zeros
    if (ieee_expt >= 0) and (ieee_expt < 254):
        ieee_frac = ibm_frac
        return np.int32(ieee_sign + (ieee_expt << 23) + ieee_frac).view(np.float32)
    elif (ieee_expt >= 254):
        return np.int32(ieee_sign + 0x7f800000).view(np.float32)
    elif (ieee_expt >= -32):
        mask = ~(0xfffffffd << (-1 - ieee_expt))
        round_up = int((ibm_frac & mask) > 0)
        ieee_frac = ((ibm_frac >> (-1 - ieee_expt)) + round_up) >> 1
        return np.int32(ieee_sign + ieee_frac).view(np.float32)
    else:
        return  np.int32(ieee_sign).view(np.float32)


@vectorize("float32(uint32)", nopython=True, cache=True)
def _numba_ibm2ieee32_vector(ibm_array):  # pragma: no cover
    """Wrapper for vectorizing IBM to IEEE conversion to arrays."""
    return _numba_ibm2ieee32_single(ibm_array)


def ieee2ibm32(ieee, endian):
    """
    Convert IEEE floating point numbers to IBM format.

    Parameters
    ----------
    ieee : np.float32
        The IEEE float(s) to convert.
    endian : char
        Output endianess: ">" for big endian, "<" for little endian.

    Returns
    -------
    np.uint32
        IBM float or array of IBM floats as np.uint32.
    """
    return _numba_ieee2ibm32_vector(ieee).astype(f"{endian}u4")


# IBM/IEEE conversion bit masks
_EXPOMASK = np.uint32(0x7f800000)
_SIGNMASK = np.uint32(0x80000000)
_MANTMASK = np.uint32(0x7fffff)


@jit("uint32(float32)", nopython=True, cache=True)
def _numba_ieee2ibm32_single(ieee):
    ieee = np.float32(ieee).view(np.uint32)
    sign = ieee & _SIGNMASK
    if ieee in [0, 2147483648]:
        return np.uint32(sign | 0x00000000)
    expo = ((ieee & _EXPOMASK) >> 23) - 127
    expo, expo_remain = divmod(expo + 1, 4)
    expo += expo_remain != 0
    downshift = 4 - expo_remain if expo_remain else 0
    expo = expo + 64
    expo = 0 if expo < 0 else expo
    expo = 127 if expo > 127 else expo
    expo = expo << 24
    expo = expo if ieee else 0
    mant = ((ieee & _MANTMASK) | 0x800000) >> downshift
    return sign | expo | mant


@vectorize("uint32(float32)", nopython=True, cache=True)
def _numba_ieee2ibm32_vector(ieee_array):  # pragma: no cover
    """Wrapper for vectorizing IEEE to IBM conversion to arrays."""
    return _numba_ieee2ibm32_single(ieee_array)
