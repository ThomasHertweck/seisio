"""Conversion of IBM floats to IEEE floats and vice versa."""

import numpy as np

# IBM/IEEE conversion bit masks
_EXPMASK = 0x7f800000
_SIGNMASK = 0x80000000
_MANTMASK = 0x7fffff


def ibm2ieee32(ibm, endian):
    """
    Convert IBM floating point numbers to IEEE format.

    If the 'ibm2ieee' package is installed (recommended), it will be used;
    this is quite fast. Otherwise a slower Numpy-based conversion is used.

    Note: Local conversion code by Robert Kern, 2011.

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
    try:
        from ibm2ieee import ibm2float32
        return ibm2float32(ibm).astype(f"{endian}f")
    except ImportError:
        sign = np.int32(ibm >> 31 & 0x01)
        exponent = np.int32(ibm >> 24 & 0x7f)
        mantissa = np.float64(ibm & 0x00ffffff)/16777216
        return ((1-2*sign)*mantissa*np.power(16.0, exponent-64)).astype(f"{endian}f")


def ieee2ibm32(ieee, endian):
    """
    Convert IEEE floating point numbers to IBM format.

    Note: Local conversion code by Robert Kern, 2011.

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
    # ieee = ieee.astype(np.float32)
    asint = ieee.view('i4')
    signbit = asint & _SIGNMASK
    exponent = ((asint & _EXPMASK) >> 23) - 127
    # The IBM 7-bit exponent is to the base 16 and the mantissa is presumed to
    # be entirely to the right of the radix point. In contrast, the IEEE
    # exponent is to the base 2 and there is an assumed 1-bit to the left of
    # the radix point.
    exp16 = ((exponent+1)//4)
    exp_remainder = (exponent+1) % 4
    exp16 += exp_remainder != 0
    downshift = np.where(exp_remainder, 4-exp_remainder, 0)
    ibm_exponent = np.clip(exp16 + 64, 0, 127)
    expbits = ibm_exponent << 24
    # Add the implicit initial 1-bit to the 23-bit IEEE mantissa to get the
    # 24-bit IBM mantissa. Downshift it by the remainder from the exponent's
    # division by 4. It is allowed to have up to 3 leading 0s.
    ibm_mantissa = ((asint & _MANTMASK) | 0x800000) >> downshift
    # Special-case: 0.0
    ibm_mantissa = np.where(ieee, ibm_mantissa, 0)
    expbits = np.where(ieee, expbits, 0)
    return (signbit | expbits | ibm_mantissa).astype(f"{endian}u4")
