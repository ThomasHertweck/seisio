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
def _numba_ibm2ieee32_vector(ibm_array):
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
def _numba_ieee2ibm32_vector(ieee_array):
    """Wrapper for vectorizing IEEE to IBM conversion to arrays."""
    return _numba_ieee2ibm32_single(ieee_array)


###############################################################################
# Test
###############################################################################

if __name__ == "__main__":
    single_to_single_pairs_ibm2ieee = [
        (0x00000000, 0.0),
        (0x00000001, 0.0),
        (0x3F000000, 0.0),
        (0x7F000000, 0.0),
        (0x1B100000, 0.0),
        (0x1B200000, 0.0),
        (0x1B400000, 0.0),
        (0x1B400001, float.fromhex("0x1p-149")),
        (0x1B800000, float.fromhex("0x1p-149")),
        (0x1BBFFFFF, float.fromhex("0x1p-149")),
        (0x1BC00000, float.fromhex("0x2p-149")),
        # Checking round-ties-to-even behaviour on a mid-range subnormal
        (0x1DA7BFFF, float.fromhex("0x14fp-149")),
        (0x1DA7C000, float.fromhex("0x150p-149")),
        (0x1DA84000, float.fromhex("0x150p-149")),
        (0x1DA84001, float.fromhex("0x151p-149")),
        (0x1DA8BFFF, float.fromhex("0x151p-149")),
        (0x1DA8C000, float.fromhex("0x152p-149")),
        (0x1DA94000, float.fromhex("0x152p-149")),
        (0x1DA94001, float.fromhex("0x153p-149")),
        (0x1DA9BFFF, float.fromhex("0x153p-149")),
        (0x1DA9C000, float.fromhex("0x154p-149")),
        (0x1DAA4000, float.fromhex("0x154p-149")),
        (0x1DAA4001, float.fromhex("0x155p-149")),
        (0x1FFFFFFF, float.fromhex("0x1p-132")),
        (0x20FFFFF4, float.fromhex("0x0.fffff0p-128")),
        (0x20FFFFF5, float.fromhex("0x0.fffff8p-128")),
        (0x20FFFFF6, float.fromhex("0x0.fffff8p-128")),
        (0x20FFFFF7, float.fromhex("0x0.fffff8p-128")),
        (0x20FFFFF8, float.fromhex("0x0.fffff8p-128")),
        (0x20FFFFF9, float.fromhex("0x0.fffff8p-128")),
        (0x20FFFFFA, float.fromhex("0x0.fffff8p-128")),
        (0x20FFFFFB, float.fromhex("0x0.fffff8p-128")),
        (0x20FFFFFC, float.fromhex("0x1p-128")),
        (0x20FFFFFD, float.fromhex("0x1p-128")),
        (0x20FFFFFE, float.fromhex("0x1p-128")),
        (0x20FFFFFF, float.fromhex("0x1p-128")),  # largest rounded case
        (0x21100000, float.fromhex("0x1p-128")),
        (0x21200000, float.fromhex("0x1p-127")),
        (0x213FFFFF, float.fromhex("0x0.fffffcp-126")),
        (0x21400000, float.fromhex("0x1p-126")),  # smallest positive normal
        (0x40800000, 0.5),
        (0x46000001, 1.0),
        (0x45000010, 1.0),
        (0x44000100, 1.0),
        (0x43001000, 1.0),
        (0x42010000, 1.0),
        (0x41100000, 1.0),
        (0x41200000, 2.0),
        (0x41300000, 3.0),
        (0x41400000, 4.0),
        (0x41800000, 8.0),
        # Test full range of possible leading zero counts.
        (0x48000001, float.fromhex("0x1p+8")),
        (0x48000002, float.fromhex("0x1p+9")),
        (0x48000004, float.fromhex("0x1p+10")),
        (0x48000008, float.fromhex("0x1p+11")),
        (0x48000010, float.fromhex("0x1p+12")),
        (0x48000020, float.fromhex("0x1p+13")),
        (0x48000040, float.fromhex("0x1p+14")),
        (0x48000080, float.fromhex("0x1p+15")),
        (0x48000100, float.fromhex("0x1p+16")),
        (0x48000200, float.fromhex("0x1p+17")),
        (0x48000400, float.fromhex("0x1p+18")),
        (0x48000800, float.fromhex("0x1p+19")),
        (0x48001000, float.fromhex("0x1p+20")),
        (0x48002000, float.fromhex("0x1p+21")),
        (0x48004000, float.fromhex("0x1p+22")),
        (0x48008000, float.fromhex("0x1p+23")),
        (0x48010000, float.fromhex("0x1p+24")),
        (0x48020000, float.fromhex("0x1p+25")),
        (0x48040000, float.fromhex("0x1p+26")),
        (0x48080000, float.fromhex("0x1p+27")),
        (0x48100000, float.fromhex("0x1p+28")),
        (0x48200000, float.fromhex("0x1p+29")),
        (0x48400000, float.fromhex("0x1p+30")),
        (0x48800000, float.fromhex("0x1p+31")),
        (0x60FFFFFF, float.fromhex("0x0.ffffffp+128")),
        (0x61100000, float("inf")),
        (0x61200000, float("inf")),
        (0x61400000, float("inf")),
        (0x62100000, float("inf")),
        (0x7FFFFFFF, float("inf")),
        # From https://en.wikipedia.org/wiki/IBM_hexadecimal_floating_point
        (0b11000010011101101010000000000000, -118.625),
    ]

    import unittest
    TC = unittest.TestCase()

    def assertFloatsIdentical(a, b):
        TC.assertEqual((a, np.signbit(a)), (b, np.signbit(b)))

    def test_single_to_single_ibm2ieee():
        print("IBM2IEEE conversion single-to-single")
        for inp, expected in single_to_single_pairs_ibm2ieee:
            pos_input = np.uint32(inp)
            pos_expected = np.float32(expected)
            pos_result = ibm2ieee32(pos_input, "<")
            print(f"(1) pos_result = {pos_result}, pos_expected = {pos_expected}")
            assertFloatsIdentical(pos_result, pos_expected)
            neg_input = np.uint32(inp ^ 0x80000000)
            neg_expected = -np.float32(expected)
            neg_result = ibm2ieee32(neg_input, "<")
            print(f"(2) neg_result = {neg_result}, neg_expected = {neg_expected}")
            assertFloatsIdentical(neg_result, neg_expected)
        return

    test_single_to_single_ibm2ieee()

    def test_vector_to_vector_ibm2ieee():
        nlist = len(single_to_single_pairs_ibm2ieee)
        pos_input = np.empty((nlist,), dtype=np.uint32)
        pos_expected = np.empty((nlist,), dtype=np.float32)
        neg_input = np.empty((nlist,), dtype=np.uint32)
        neg_expected = np.empty((nlist,), dtype=np.float32)
        ii = 0
        for inp, expected in single_to_single_pairs_ibm2ieee:
            pos_input[ii] = np.uint32(inp)
            pos_expected[ii] = np.float32(expected)
            neg_input[ii] = np.uint32(inp ^ 0x80000000)
            neg_expected[ii] = -np.float32(expected)
            ii += 1

        print("IBM2IEEE conversion vector-to-vector")
        pos_result = ibm2ieee32(pos_input, "<")
        neg_result = ibm2ieee32(neg_input, "<")

        for i in np.arange(nlist):
            print(f"(1) pos_result = {pos_result[i]}, pos_expected = {pos_expected[i]}")
            assertFloatsIdentical(pos_result[i], pos_expected[i])
            print(f"(2) neg_result = {neg_result[i]}, neg_expected = {neg_expected[i]}")
            assertFloatsIdentical(neg_result[i], neg_expected[i])
        return

    test_vector_to_vector_ibm2ieee()

    single_to_single_pairs_ieee2ibm = [
        (-118.625, 0b11000010011101101010000000000000),
        (0.0, 0b00000000000000000000000000000000),
        (300.0, 0x4312C000),
        (0.5, 0x40800000),
        (1.0, 0x41100000),
        (2.0, 0x41200000),
        (3.0, 0x41300000),
        (4.0, 0x41400000),
        (8.0, 0x41800000),
        (float.fromhex("0x0.ffffffp+128"), 0x60FFFFFF),
        ]

    def test_vector_to_vector_ieee2ibm():
        # Inputs with known outputs.
        nlist = len(single_to_single_pairs_ieee2ibm)
        pos_input = np.empty((nlist,), dtype=np.float32)
        pos_expected = np.empty((nlist,), dtype=np.uint32)
        neg_input = np.empty((nlist,), dtype=np.float32)
        neg_expected = np.empty((nlist,), dtype=np.uint32)
        ii = 0
        for inp, expected in single_to_single_pairs_ieee2ibm:
            pos_input[ii] = np.float32(inp)
            pos_expected[ii] = np.uint32(expected)
            neg_input[ii] = -np.float32(inp)
            neg_expected[ii] = np.uint32(expected ^ 0x80000000)
            ii += 1

        print("IEEE2IBM conversion vector-to-vector")
        pos_result = ieee2ibm32(pos_input, "<")
        neg_result = ieee2ibm32(neg_input, "<")

        for i in np.arange(nlist):
            print(f"(1) pos_result = {pos_result[i]}, pos_expected = {pos_expected[i]}")
            assertFloatsIdentical(pos_result[i], pos_expected[i])
            print(f"(2) neg_result = {neg_result[i]}, neg_expected = {neg_expected[i]}")
            assertFloatsIdentical(neg_result[i], neg_expected[i])

        print("IEEE2IBM and IBM2IEEE roundtrip vector-to-vector")
        pos_result = ieee2ibm32(pos_input, "<")
        pos_input_inv = ibm2ieee32(pos_result, "<")
        neg_result = ieee2ibm32(neg_input, "<")
        neg_input_inv = ibm2ieee32(neg_result, "<")

        for i in np.arange(nlist):
            print(f"(1) pos_input = {pos_input[i]}, pos_recovered = {pos_input_inv[i]}")
            assertFloatsIdentical(pos_input[i], pos_input_inv[i])
            print(f"(2) neg_input = {neg_input[i]}, neg_recovered = {neg_input_inv[i]}")
            assertFloatsIdentical(neg_input[i], neg_input_inv[i])
        return

    test_vector_to_vector_ieee2ibm()

#(1) pos_result = 1125122048, pos_expected = 1207959553
#(2) neg_result = 3272605696, neg_expected = 3355443201
