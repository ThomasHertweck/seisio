import numpy as np
import pytest
import struct

from pathlib import Path


TESTS_DIR = Path(__file__).parent.resolve()


def create_trace_header(header_dict, endian="<"):
    """
    Creates a 240-byte trace header initialized to zeros,
    with specific byte offsets set to custom integer values.

    overrides: dict (default: None)
        Mapping byte_offset to (value, format_type)
    """
    header = bytearray(240)
    for offset, (val, fmt) in header_dict.items():
        packed = struct.pack(f"{endian}{fmt}", val)
        header[offset : offset + len(packed)] = packed
    return bytes(header)


@pytest.fixture(params=["<", ">"])
def dummy_su_file(tmp_path, request):
    endian = request.param
    if endian == "<":
        end = "little"
    else:
        end = "big"
    file_path = tmp_path / f"dummy_{end}.su"

    nt = 12
    ns = 20

    data = np.zeros((nt, ns), dtype=f"{endian}f")
    for i in range(nt):
        data[i, :] = i*1000 + np.arange(0, ns, 1)

    header_values = {
        0:   (1, 'i'),     # tracl = 1 (4-byte int at byte 0)
        4:   (99, 'i'),    # tracr = 99 (4-byte int at byte 4)
        8:   (1, 'i'),     # fldr = 1 (4-byte int at byte 8)
        12:  (1, 'i'),     # tracf = 1 (4-byte int at byte 12)
        20:  (1, 'i'),     # cdp = 1 (4-byte int at byte 20)
        24:  (1, 'i'),     # cdpt = 1 (4-byte int at byte 24)
        28:  (1, 'h'),     # trid = 1 (2-byte short at byte 38)
        36:  (1, 'i'),     # offset = 1 (4-byte int at byte 36)
        108: (42, 'h'),    # delrt = 10 (2-byte short at byte 108)
        114: (ns, 'h'),    # ns = ns samples (2-byte short at byte 114)
        116: (2000, 'h')   # dt = 2000 microseconds / 2ms (2-byte short at byte 116)
    }

    with open(file_path, "wb") as f:
        fldr = 0
        cdp = 0
        tracf = 0
        cdpt = 0
        offset = 0
        for i in range(nt):
            header_values[0] = (1+i, 'i')
            if i%4 == 0:
                fldr += 1
                cdp += 2
                tracf = 1
                cdpt = 1
                offset = 0
            else:
                tracf += 1
                cdpt += 1
                offset += 500
            header_values[8] = (fldr, 'i')
            header_values[12] = (tracf, 'i')
            header_values[20] = (cdp, 'i')
            header_values[24] = (cdpt, 'i')
            header_values[36] = (offset, 'i')
            header_bytes = create_trace_header(header_values, endian=endian)
            f.write(header_bytes)
            f.write(data[i, :].tobytes())

    return str(file_path)


@pytest.fixture(params=["<", ">"])
def dummy_special_su_file(tmp_path, request):
    endian = request.param
    if endian == "<":
        end = "little"
    else:
        end = "big"
    file_path = tmp_path / f"dummy_{end}.su"

    nt = 1
    ns = 2

    data = np.zeros((nt, ns), dtype=f"{endian}f")
    for i in range(nt):
        data[i, :] = np.zeros((ns,))

    header_values = {
        108: (42, 'h'),    # delrt = 10 (2-byte short at byte 108)
        114: (ns, 'h'),    # ns = ns samples (2-byte short at byte 114)
        116: (20560, 'h')   # dt = 20560 microseconds  (2-byte short at byte 116)
    }

    with open(file_path, "wb") as f:
        for i in range(nt):
            header_bytes = create_trace_header(header_values, endian=endian)
            f.write(header_bytes)
            f.write(data[i, :].tobytes())

    return str(file_path)

_EXPOMASK = np.uint32(0x7f800000)
_SIGNMASK = np.uint32(0x80000000)
_MANTMASK = np.uint32(0x7fffff)

def _ieee2ibm_single(ieee):
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

def _ieee2ibm(ieee_array, endian):
    flat = ieee_array.astype(np.float32).ravel()
    out = np.zeros(len(flat), dtype=np.uint32)
    for i, val in enumerate(flat):
        out[i] = _ieee2ibm_single(val)
    return out.astype(f"{endian}u4")


@pytest.fixture(
    params=[
        # (endian, format_code, text_encoding, num_ext_text, num_ext_trace_headers, add_trailer)
        (">", 1, "ebcdic", 0, 0, False),
        (">", 1, "ascii", 0, 0, False),
        ("<", 5, "ebcdic", 0, 0, False),
        ("<", 5, "ascii", 0, 0, False),
        (">", 5, "ebcdic", 0, 0, False),
        (">", 5, "ebcdic", 2, 1, True),
        (">", 3, "ebcdic", 0, 0, False),
        (">", 6, "ebcdic", 0, 0, False),
        (">", 8, "ebcdic", 0, 0, False),
        (">", 9, "ebcdic", 0, 0, False),
        (">", 10, "ebcdic", 0, 0, False),
        (">", 11, "ebcdic", 0, 0, False),
        (">", 12, "ebcdic", 0, 0, False),
        (">", 16, "ebcdic", 0, 0, False),
    ]
)
def dummy_segy_file(tmp_path, request):
    endian, data_format, text_encoding, num_ext_text, num_ext_trace_headers, add_trailer = request.param

    file_path = tmp_path / f"dummy_{'little' if endian == '<' else 'big'}_fmt{data_format}_{text_encoding}_{num_ext_text}_{num_ext_trace_headers}_{1 if add_trailer else 0}.sgy"

    nt = 12
    ns = 20

    data = np.zeros((nt, ns), dtype=np.float32)
    for i in range(nt):
        data[i, :] = i * 1000 + np.arange(0, ns, 1)

    with open(file_path, "wb") as f:
        # ====================================================================
        # 1. primary textual file header (3200 bytes)
        # ====================================================================
        lines = [f"C{i+1:02d} " + f"SEG-Y DUMMY FILE HEADER LINE {i+1}".ljust(76) for i in range(40)]
        text_header_str = "".join(lines)
        if text_encoding.lower() == "ebcdic":
            text_bytes = text_header_str.encode("cp500")  # EBCDIC encoding
        else:
            text_bytes = text_header_str.encode("ascii")
        f.write(text_bytes)

        # ====================================================================
        # 2. binary file header (400 bytes)
        # ====================================================================
        if num_ext_text or num_ext_trace_headers or  add_trailer:
            major = 2
        else:
            if data_format in [8, 16]:
                major = 0
            else:
                major = 1
        bin_header = bytearray(400)
        struct.pack_into(f"{endian}h", bin_header, 16, 2000)
        struct.pack_into(f"{endian}h", bin_header, 18, 2000)
        struct.pack_into(f"{endian}h", bin_header, 20, ns)
        struct.pack_into(f"{endian}h", bin_header, 22, ns)
        struct.pack_into(f"{endian}h", bin_header, 24, data_format)
        if data_format not in [8, 16]:
            struct.pack_into(f"{endian}I", bin_header, 96, 16909060)
        struct.pack_into(f"{endian}B", bin_header, 300, major)
        if data_format not in [8, 16]:
            struct.pack_into(f"{endian}h", bin_header, 302, 1)
        elif data_format == 12:
            struct.pack_into(f"{endian}h", bin_header, 302, 3)
        struct.pack_into(f"{endian}h", bin_header, 304, num_ext_text)
        struct.pack_into(f"{endian}i", bin_header, 306, num_ext_trace_headers)
        struct.pack_into(f"{endian}i", bin_header, 328, 1 if add_trailer else 0)
        f.write(bin_header)

        # ====================================================================
        # 3. extended textual headers (SEG-Y Rev 1/2) (3200 bytes * num_ext_text)
        # ====================================================================
        for ext_idx in range(num_ext_text):
            ext_lines = [f"K{i+1:02d} EXTENDED HEADER {ext_idx+1} LINE {i+1}".ljust(80) for i in range(40)]
            ext_str = "".join(ext_lines)
            f.write(ext_str.encode("cp500" if text_encoding == "ebcdic" else "ascii"))

        # ====================================================================
        # 4. traces (trace header +extension headers + data)
        # ====================================================================
        fldr = 0
        cdp = 0
        iline = 100
        xline = 201
        for i in range(nt):
            if i % 4 == 0:
                fldr += 1
                iline += 1
                xline = 201
                cdp += 2
                tracf = 1
                cdpt = 1
                offset = 0
            else:
                xline += 1
                tracf += 1
                cdpt += 1
                offset += 500

            # Standard 240-byte Trace Header
            header_values = {
                0: (i + 1, "i"),              # Byte 1-4: tracl
                4: (99, "i"),                 # Byte 5-8: tracr
                8: (fldr, "i"),               # Byte 9-12: fldr
                12: (tracf, "i"),             # Byte 13-16: tracf
                20: (cdp, "i"),               # Byte 21-24: cdp
                24: (cdpt, "i"),              # Byte 25-28: cdpt
                28: (1, "h"),                 # Byte 29-30: trid
                36: (offset, "i"),            # Byte 37-40: offset
                108: (10, "h"),               # Byte 109-110: delrt
                114: (ns, "h"),               # Byte 115-116: ns
                116: (2000, "h"),             # Byte 117-118: dt
                156: (num_ext_trace_headers, "h"),  # Byte 157-158: number of Trace Header Extensions
                188: (iline, "i"),            # Byte 189-192: iline
                192: (xline, "i"),            # Byte 193-196: xline
            }
            f.write(create_trace_header(header_values, endian=endian))

            # trace header extensions (240 bytes each)
            for _ in range(num_ext_trace_headers):
                ext_trace_hdr = bytearray(240)
                struct.pack_into(f"{endian}i", ext_trace_hdr, 0, 1) # Sample extended metadata
                f.write(ext_trace_hdr)

            # sample data packing based on format code
            trace_data = data[i, :]
            if data_format == 1:    # 4-byte IBM Float
                ibm_bytes = _ieee2ibm(trace_data, endian).tobytes()
                f.write(ibm_bytes)
            elif data_format == 2:  # 4-byte Signed Integer
                f.write(trace_data.astype(f"{endian}i").tobytes())
            elif data_format == 3:  # 2-byte Signed Integer
                f.write(trace_data.astype(f"{endian}h").tobytes())
            elif data_format == 5:  # 4-byte IEEE Float
                f.write(trace_data.astype(f"{endian}f").tobytes())
            elif data_format == 6:  # 8-byte IEEE Float
                f.write(trace_data.astype(f"{endian}d").tobytes())
            elif data_format == 8:  # 1-byte Signed Char
                f.write(trace_data.astype("i1").tobytes())
            elif data_format == 9:  # 8-byte Signed Integer
                f.write(trace_data.astype(f"{endian}q").tobytes())
            elif data_format == 10:  # 4-byte Unsigned Integer
                f.write(trace_data.astype(f"{endian}I").tobytes())
            elif data_format == 11:  # 2-byte Unsigned Integer
                f.write(trace_data.astype(f"{endian}H").tobytes())
            elif data_format == 12:  # 8-byte Unsigned Integer
                f.write(trace_data.astype(f"{endian}Q").tobytes())
            elif data_format == 16:  # 1-byte Unsigned Integer
                f.write(trace_data.astype(f"{endian}B").tobytes())
            else:
                raise ValueError(f"Unsupported SEG-Y format code: {data_format}")

        # ====================================================================
        # 5. SEG-Y revision 2 file trailer (optional)
        # ====================================================================
        if add_trailer:
            trailer_bytes = ("SEG-Y REV2 TRAILER".ljust(3200)).encode("ascii")
            f.write(trailer_bytes)

    return str(file_path)


@pytest.fixture(
    params=[
        # (endian, format_code, text_encoding, num_ext_text, num_ext_trace_headers, add_trailer)
         ("<", 5, "ascii", 2, 2, True),
         ("<", 5, "ebcdic", -1, 1, True),
         (">", 5, "ebcdic", 3, 2, True),
         (">", 5, "ascii", 0, 1, True),
    ]
)
def dummy_segy_file_special(tmp_path, request):
    endian, data_format, text_encoding, num_ext_text, num_ext_trace_headers, add_trailer = request.param

    file_path = tmp_path / f"dummy_{'little' if endian == '<' else 'big'}_fmt{data_format}_{text_encoding}_{num_ext_text}_{num_ext_trace_headers}_{1 if add_trailer else 0}.sgy"

    nt = 12
    ns = 20

    data = np.zeros((nt, ns), dtype=np.float32)
    for i in range(nt):
        data[i, :] = i * 1000 + np.arange(0, ns, 1)

    with open(file_path, "wb") as f:
        # ====================================================================
        # 1. primary textual file header (3200 bytes)
        # ====================================================================
        lines = [f"C{i+1:02d} " + f"SEG-Y DUMMY FILE HEADER LINE {i+1}".ljust(76) for i in range(40)]
        text_header_str = "".join(lines)
        if text_encoding.lower() == "ebcdic":
            text_bytes = text_header_str.encode("cp500")  # EBCDIC encoding
        else:
            text_bytes = text_header_str.encode("ascii")
        f.write(text_bytes)

        # ====================================================================
        # 2. binary file header (400 bytes)
        # ====================================================================
        if num_ext_text or num_ext_trace_headers or  add_trailer:
            major = 2
        else:
            if data_format in [8, 16]:
                major = 0
            else:
                major = 1
        bin_header = bytearray(400)
        struct.pack_into(f"{endian}h", bin_header, 16, 2000)
        struct.pack_into(f"{endian}h", bin_header, 18, 2000)
        struct.pack_into(f"{endian}h", bin_header, 20, ns)
        struct.pack_into(f"{endian}h", bin_header, 22, ns)
        struct.pack_into(f"{endian}h", bin_header, 24, data_format)
        struct.pack_into(f"{endian}I", bin_header, 96, 16909060)
        struct.pack_into(f"{endian}B", bin_header, 300, major)
        struct.pack_into(f"{endian}h", bin_header, 302, 1)
        struct.pack_into(f"{endian}h", bin_header, 304, num_ext_text)
        struct.pack_into(f"{endian}i", bin_header, 306, num_ext_trace_headers)
        if text_encoding == "ascii":
            struct.pack_into(f"{endian}i", bin_header, 328, -1)  # undefined no.
        else:
            struct.pack_into(f"{endian}i", bin_header, 328, 1 if add_trailer else 0)
        f.write(bin_header)

        # ====================================================================
        # 3. extended textual headers (SEG-Y Rev 1/2) (3200 bytes * num_ext_text)
        # ====================================================================
        if num_ext_text == -1:
              num_ext_text = 2
        for ext_idx in range(num_ext_text):
            ext_lines = [f"K{i+1:02d} EXTENDED HEADER {ext_idx+1} LINE {i+1}".ljust(80) for i in range(40)]
            ext_str = "".join(ext_lines)
            f.write(ext_str.encode("cp500" if text_encoding == "ebcdic" else "ascii"))

        # ====================================================================
        # 4. traces (trace header + extension headers + data)
        # ====================================================================
        fldr = 0
        cdp = 0
        iline = 100
        xline = 201
        for i in range(nt):
            if i % 4 == 0:
                fldr += 1
                iline += 1
                xline = 201
                cdp += 2
                tracf = 1
                cdpt = 1
                offset = 0
            else:
                xline += 1
                tracf += 1
                cdpt += 1
                offset += 500

            # Standard 240-byte Trace Header
            header_values = {
                0: (i + 1, "i"),              # Byte 1-4: tracl
                4: (99, "i"),                 # Byte 5-8: tracr
                8: (fldr, "i"),               # Byte 9-12: fldr
                12: (tracf, "i"),             # Byte 13-16: tracf
                20: (cdp, "i"),               # Byte 21-24: cdp
                24: (cdpt, "i"),              # Byte 25-28: cdpt
                28: (1, "h"),                 # Byte 29-30: trid
                36: (offset, "i"),            # Byte 37-40: offset
                108: (10, "h"),               # Byte 109-110: delrt
                114: (ns, "h"),               # Byte 115-116: ns
                116: (2000, "h"),             # Byte 117-118: dt
                156: (num_ext_trace_headers, "h"),  # Byte 157-158: number of Trace Header Extensions
                188: (iline, "i"),            # Byte 189-192: iline
                192: (xline, "i"),            # Byte 193-196: xline
            }
            f.write(create_trace_header(header_values, endian=endian))

            # trace header extensions (trace header extension 1 + user-defined trace headeers)
            for _ in range(num_ext_trace_headers):
                ext_trace_hdr = bytearray(240)
                struct.pack_into(f"{endian}i", ext_trace_hdr, 0, 1) # Sample extended metadata
                f.write(ext_trace_hdr)

            # sample data packing based on format code
            trace_data = data[i, :]
            if data_format == 1:    # 4-byte IBM Float
                ibm_bytes = _ieee2ibm(trace_data, endian).tobytes()
                f.write(ibm_bytes)
            elif data_format == 2:  # 4-byte Signed Integer
                f.write(trace_data.astype(f"{endian}i").tobytes())
            elif data_format == 3:  # 2-byte Signed Integer
                f.write(trace_data.astype(f"{endian}h").tobytes())
            elif data_format == 5:  # 4-byte IEEE Float
                f.write(trace_data.astype(f"{endian}f").tobytes())
            elif data_format == 6:  # 8-byte IEEE Float
                f.write(trace_data.astype(f"{endian}d").tobytes())
            elif data_format == 8:  # 1-byte Signed Char
                f.write(trace_data.astype("i1").tobytes())
            elif data_format == 9:  # 8-byte Signed Integer
                f.write(trace_data.astype(f"{endian}q").tobytes())
            elif data_format == 10:  # 4-byte Unsigned Integer
                f.write(trace_data.astype(f"{endian}I").tobytes())
            elif data_format == 11:  # 2-byte Unsigned Integer
                f.write(trace_data.astype(f"{endian}H").tobytes())
            elif data_format == 12:  # 8-byte Unsigned Integer
                f.write(trace_data.astype(f"{endian}Q").tobytes())
            elif data_format == 16:  # 1-byte Unsigned Integer
                f.write(trace_data.astype(f"{endian}B").tobytes())
            else:
                raise ValueError(f"Unsupported SEG-Y format code: {data_format}")

        # ====================================================================
        # 5. SEG-Y revision 2 file trailer (optional)
        # ====================================================================
        if add_trailer:
            trailer_bytes = ("((SEG:User Data)) ... SEG-Y REV2 TRAILER ... ((SEG: EndText))".ljust(3200)).encode("ascii")
            f.write(trailer_bytes)

    return str(file_path)


@pytest.fixture
def sample_seg2_file():
    """Returns the absolute path to the sample SEG2 test file."""
    file_path = TESTS_DIR / "data" / "sample.seg2"
    # fail fast if file was moved or omitted
    if not file_path.exists():
        pytest.fail(f"Test SEG2 file missing at: {file_path}")
    return file_path
