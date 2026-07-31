
import numpy as np
import pytest

from seisio import _ibm2ieee as i2i

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

single_to_single_pairs_ieee2ibm = [
    (-118.625, 0b11000010011101101010000000000000),
    (0.0, 0b00000000000000000000000000000000),
    (300.0, 0x4312C000),
    (1.0, 0x41100000),
    (2.0, 0x41200000),
    (3.0, 0x41300000),
    (4.0, 0x41400000),
    (8.0, 0x41800000),
    (float.fromhex("0x0.ffffffp+128"), 0x60FFFFFF),
    ]

@pytest.mark.parametrize("inp, ref", single_to_single_pairs_ibm2ieee)
def test_single_to_single_ibm2ieee(inp, ref):
    pos_input = np.uint32(inp)
    pos_expected = np.float32(ref)
    pos_result = i2i.ibm2ieee32(pos_input, "<")
    assert pos_result == pos_expected
    pos_result = i2i._numba_ibm2ieee32_single(pos_input)
    assert pos_result == pos_expected

    neg_input = np.uint32(inp ^ 0x80000000)
    neg_expected = -np.float32(ref)
    neg_result = i2i.ibm2ieee32(neg_input, "<")
    assert neg_result == neg_expected
    neg_result =  i2i._numba_ibm2ieee32_single(neg_input)
    assert neg_result == neg_expected

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

    pos_result = i2i.ibm2ieee32(pos_input, "<")
    neg_result = i2i.ibm2ieee32(neg_input, "<")
    for i in np.arange(nlist):
        assert pos_result[i] == pos_expected[i]
        assert neg_result[i] == neg_expected[i]
    pos_result = i2i._numba_ibm2ieee32_vector(pos_input)
    neg_result = i2i._numba_ibm2ieee32_vector(neg_input)
    for i in np.arange(nlist):
        assert pos_result[i] == pos_expected[i]
        assert neg_result[i] == neg_expected[i]

@pytest.mark.parametrize("inp, ref", single_to_single_pairs_ieee2ibm)
def test_single_to_single_ieee2ibm(inp, ref):
    pos_input = np.float32(inp)
    pos_expected = np.uint32(ref)
    pos_result = i2i.ieee2ibm32(pos_input, "<")
    assert pos_result == pos_expected
    pos_result = i2i._numba_ieee2ibm32_single(pos_input)
    assert pos_result == pos_expected

    neg_input = -np.float32(inp)
    neg_expected = np.uint32(ref ^ 0x80000000)
    neg_result = i2i.ieee2ibm32(neg_input, "<")
    assert neg_result == neg_expected
    neg_result = i2i._numba_ieee2ibm32_single(neg_input)
    assert neg_result == neg_expected

def test_vector_to_vector_iee2ibm():
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

    pos_result = i2i.ieee2ibm32(pos_input, "<")
    neg_result = i2i.ieee2ibm32(neg_input, "<")
    for i in np.arange(nlist):
        assert pos_result[i] == pos_expected[i]
        assert neg_result[i] == neg_expected[i]
    pos_result = i2i._numba_ieee2ibm32_vector(pos_input)
    neg_result = i2i._numba_ieee2ibm32_vector(neg_input)
    for i in np.arange(nlist):
        assert pos_result[i] == pos_expected[i]
        assert neg_result[i] == neg_expected[i]
