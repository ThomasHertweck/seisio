
import pytest

from seisio import _txtheader as txh

def test_wrong_encoding():
    with pytest.raises(ValueError):
        txh.TxtHeader(encoding="StrangeValue")
    with pytest.raises(TypeError):
        txh.TxtHeader(encoding=42.0)

def test_encoding():
    t = txh.TxtHeader(encoding="ascii")
    assert t is not None
    t = txh.TxtHeader(encoding="AsCiI")
    assert t is not None
    t = txh.TxtHeader(encoding="ebcdic")
    assert t is not None

def test_no_encoding():
    t = txh.TxtHeader()
    assert t is not None
    enc = t.encoding
    assert enc is None

def test_template():
    t = txh.TxtHeader(encoding="ascii")
    tmp = t.template()
    assert tmp is not None

def test_properties():
    t = txh.TxtHeader(encoding="ascii")
    assert len(t) == 3200
    info = t.info
    assert "SEG-Y textual file header" in info
    enc = t.encoding
    assert enc == "ASCII"
    h = t.header
    assert h is not None

def test_set_properties():
    t = txh.TxtHeader(encoding="ascii")
    assert len(t) == 3200
    t.info = "My header"
    t.encoding = "EBCDIC"
    with pytest.raises(ValueError):
        t.encoding = "unknown"
    tmp = t.template()
    t.header = tmp

def test_set_header():
    t = txh.TxtHeader(encoding="ebcdic")
    assert len(t) == 3200
    myheader = t.template()
    tmp = list(myheader[0])
    tmp[4:7] = "ABC"
    myheader[0] = "".join(tmp)
    tmp = list(myheader[37])
    tmp[4:7] = "XYZ"
    myheader[37] = "".join(tmp)
    t.header = myheader
    ebcdic = t.get_header(decode=False)
    conv = ebcdic.decode("cp037")
    asciihead = t.get_header(decode=True)
    assert conv[0] == asciihead[0]
    h = t.header
    assert h is not None

def test_set_header_special():
    t = txh.TxtHeader(encoding="ascii")
    assert len(t) == 3200
    t.header = "AAA"
    h = t.get_header(decode=True)
    assert h is not None
    t.header = ["AAA", "BBB"]
    h = t.get_header(decode=True)
    assert h is not None
    t._encoding = None
    t.header = ["AAA"]
    h = t.get_header(decode=True)
    assert h is not None
    t.set_header(h, encode=False)
    h = t.get_header(decode=False)
    assert h is not None

def test_log_txthead():
    t = txh.TxtHeader(encoding="ebcdic")
    assert len(t) == 3200
    myheader = t.template()
    tmp = list(myheader[0])
    tmp[4:7] = "ABC"
    myheader[0] = "".join(tmp)
    tmp = list(myheader[37])
    tmp[4:7] = "XYZ"
    myheader[37] = "".join(tmp)
    t.header = myheader
    t.log_txthead()

def test_log_txthead_none():
    t = txh.TxtHeader(encoding="ebcdic")
    assert len(t) == 3200
    t._list = None
    t.log_txthead()

def test_ascii_or_ebcdic():
    t = txh.TxtHeader(encoding="ebcdic")
    assert len(t) == 3200
    myheader = t.template()
    tmp = list(myheader[0])
    tmp[4:7] = "ABC"
    myheader[0] = "".join(tmp)
    tmp = list(myheader[37])
    tmp[4:7] = "XYZ"
    myheader[37] = "".join(tmp)
    t.header = myheader
    res = t._ascii_or_ebcdic()
    assert res == "cp037"
    t.encoding = "ascii"
    t.header = myheader
    res = t._ascii_or_ebcdic()
    assert res == "ascii"

def test_get_header():
    t = txh.TxtHeader(encoding="ascii")
    assert len(t) == 3200
    myheader = t.template()
    tmp = list(myheader[0])
    tmp[4:7] = "ABC"
    myheader[0] = "".join(tmp)
    tmp = list(myheader[37])
    tmp[4:7] = "XYZ"
    myheader[37] = "".join(tmp)
    t.header = myheader
    t._bytes = None
    h = t.get_header(decode=False)
    assert h == ""

def test_read_header(dummy_segy_file):
    t = txh.TxtHeader(encoding=None)
    with open(dummy_segy_file, "rb") as file:
         t.read(file)
    assert t._bytes is not None
    header =  t.get_header(decode=True)
    assert header is not None

def test_write_header(tmp_path):
    t = txh.TxtHeader(encoding="ascii")
    assert len(t) == 3200
    myheader = t.template(fill=False)
    tmp = list(myheader[0])
    tmp[4:7] = "ABC"
    myheader[0] = "".join(tmp)
    tmp = list(myheader[37])
    tmp[4:7] = "XYZ"
    myheader[37] = "".join(tmp)
    t.header = myheader
    dummy =  tmp_path / "dummy_header.segy"
    with open(dummy, "wb") as file:
        t.write(file)
    t2 = txh.TxtHeader()
    with open(dummy, "rb") as file:
         t2.read(file)
    assert t2._bytes is not None
    header =  t2.get_header(decode=True)
    assert header is not None

def test_short_read(tmp_path):
    file_path = tmp_path / "short_file"
    with open(file_path, "wb") as f:
        text_bytes = "short".encode("ascii")
        f.write(text_bytes)
    t = txh.TxtHeader(encoding="ascii")
    with pytest.raises(EOFError):
        with open(file_path, "rb") as f:
            t.read(f)
