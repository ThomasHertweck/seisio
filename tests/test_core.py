
import json
import pytest
import seisio
import sys

from importlib.resources import files

def test_version():
    assert hasattr(seisio, "__version__")

def test_author():
    assert hasattr(seisio, "__author__")

def test_copyright():
    assert hasattr(seisio, "__copyright__")

def test_license():
    assert hasattr(seisio, "__license__")

def test_segy_traceheader_json_exists():
    json_resource = files("seisio").joinpath("json", "segy_traceheader.json")
    assert json_resource.is_file(), "segy_traceheader.json is missing!"
    with json_resource.open("r", encoding="utf-8") as f:
        header_data = json.load(f)
    assert isinstance(header_data, dict), "segy_traceheader.json cannot be parsed!"

def test_segy_binaryheader_json_exists():
    json_resource = files("seisio").joinpath("json", "segy_binaryheader.json")
    assert json_resource.is_file(), "segy_binaryheader.json is missing!"
    with json_resource.open("r", encoding="utf-8") as f:
        header_data = json.load(f)
    assert isinstance(header_data, dict), "segy_binaryheader.json cannot be parsed!"

def test_segy_traceheader_ext1_json_exists():
    json_resource = files("seisio").joinpath("json", "segy_traceheader_ext1.json")
    assert json_resource.is_file(), "segy_traceheader_ext1.json is missing!"
    with json_resource.open("r", encoding="utf-8") as f:
        header_data = json.load(f)
    assert isinstance(header_data, dict), "segy_traceheader_ext1.json cannot be parsed!"

def test_su_traceheader_json_exists():
    json_resource = files("seisio").joinpath("json", "su_traceheader.json")
    assert json_resource.is_file(), "su_traceheader.json is missing!"
    with json_resource.open("r", encoding="utf-8") as f:
        header_data = json.load(f)
    assert isinstance(header_data, dict), "su_traceheader.json cannot be parsed!"

@pytest.mark.parametrize("thdef", ["segy_traceheader.json",
                                   "segy_binaryheader.json",
                                   "segy_traceheader_ext1.json",
                                   "su_traceheader.json"])
def test_check_thdef_validity(thdef):
    json_resource = files("seisio").joinpath("json", thdef)
    valid = seisio.check_thdef_validity(json_resource)
    assert valid == True, f"{thdef} is invalid!"

def test_check_thdef_validity_overlap(tmp_path):
    json_resource = tmp_path / "dummy.json"
    d = {"tracl": {"byte": 1, "type": "i", "desc": "AAA"},
         "tracr": {"byte": 3, "type": "i", "desc": "BBB"}}
    json_resource.write_text(json.dumps(d, indent=4))
    valid = seisio.check_thdef_validity(json_resource)
    assert valid == False

def test_check_thdef_validity_titles(tmp_path):
    json_resource = tmp_path / "dummy.json"
    d = {"tracl": {"byte": 1, "type": "i", "desc": "AAA"},
         "tracr": {"byte": 5, "type": "i", "desc": "AAA"}}
    json_resource.write_text(json.dumps(d, indent=4))
    valid = seisio.check_thdef_validity(json_resource)
    assert valid == True

def test_check_thdef_validity_key(tmp_path):
    json_resource = tmp_path / "dummy.json"
    d = {"tracl": {"byte": 1, "type": "i", "desc": "AAA"},
         "tracr": {"byte": 5, "typeeee": "X", "desc": "AAA"}}
    json_resource.write_text(json.dumps(d, indent=4))
    valid = seisio.check_thdef_validity(json_resource)
    assert valid == False

def test_log_sgy_default_thdef():
    lst = seisio.log_sgy_default_thdef()
    assert isinstance(lst, list)
    assert "cdp" in lst
    assert "ns" in lst
    assert "xline" in lst

def test_log_sgy_default_thdef_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "tabulate", raising=False)
    monkeypatch.setitem(sys.modules, "tabulate", None)
    lst = seisio.log_sgy_default_thdef()
    assert isinstance(lst, list)
    assert "cdp" in lst
    assert "ns" in lst
    assert "xline" in lst

def test_log_su_default_thdef():
    lst = seisio.log_su_default_thdef()
    assert isinstance(lst, list)
    assert "cdp" in lst
    assert "ns" in lst
    assert "d1" in lst

def test_log_su_default_thdef_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "tabulate", raising=False)
    monkeypatch.setitem(sys.modules, "tabulate", None)
    lst = seisio.log_su_default_thdef()
    assert isinstance(lst, list)
    assert "cdp" in lst
    assert "ns" in lst
    assert "d1" in lst

def test_segy_bhdef_template(tmp_path):
     file = tmp_path / "segy_bh.json"
     seisio.segy_bhdef_template(file)
     assert file.exists()
     valid = seisio.check_thdef_validity(file)
     assert valid == True, f"{file} is invalid!"

def test_segy_thdef1_template(tmp_path):
     file = tmp_path / "segy_th1.json"
     seisio.segy_thdef1_template(file)
     assert file.exists()
     valid = seisio.check_thdef_validity(file)
     assert valid == True, f"{file} is invalid!"

def test_segy_thdef_template(tmp_path):
     file = tmp_path / "segy_th.json"
     seisio.segy_thdef_template(file)
     assert file.exists()
     valid = seisio.check_thdef_validity(file)
     assert valid == True, f"{file} is invalid!"

def test_su_thdef_template(tmp_path):
     file = tmp_path / "su_th.json"
     seisio.su_thdef_template(file)
     assert file.exists()
     valid = seisio.check_thdef_validity(file)
     assert valid == True, f"{file} is invalid!"

@pytest.mark.parametrize("major, minor", [(0, 0),
                                          (1, 0),
                                          (2, 0),
                                          (2, 1)])
def test_segy_txthead_template(major, minor):
    lst1 = seisio.segy_txthead_template(major_version=major,
                                       minor_version=minor,
                                       fill=True)
    assert ("C01" in lst1[0]) and (len(lst1[0]) == 80)
    assert ("C40" in lst1[39]) and (len(lst1[39]) == 80)
    assert f"{major}.{minor}" in lst1[38]
    assert "END TEXTUAL HEADER" in lst1[39]

    lst2 = seisio.segy_txthead_template(major_version=major,
                                       minor_version=minor,
                                       fill=False)
    assert ("C01" not in lst2[0]) and (len(lst2[0]) == 80)
    assert ("C40" not in lst2[39]) and (len(lst2[39]) == 80)
    assert f"{major}.{minor}" not in lst2[38]
    assert "END TEXTUAL HEADER" not in lst2[39]

def test_missing_input():
    with pytest.raises(TypeError):
        seisio.input()
    with pytest.raises(FileNotFoundError):
        seisio.input("does_not_exist.sgy")
    with pytest.raises(RuntimeError):
        seisio.input("unknown_type")

@pytest.mark.parametrize("suffix", ["SGY",
                                    "SEGY",
                                    "SEG-Y",
                                    "SEG_Y",
                                    "SEG2",
                                    "DAT",
                                    "S2",
                                    "SG2",
                                    "SU"])
def test_missing_input_suffix(tmp_path, suffix):
    file = tmp_path / "dummy"
    with open(file, mode="wb") as f:
        f.truncate(8192) # write 8k
    with pytest.raises(RuntimeError):
        seisio.input(file)
    if suffix in ["SGY", "SEGY", "SEG-Y", "SEG_Y"]:
        with pytest.raises(KeyError):
            # wrong data format leads to key error
            seisio.input(file, filetype=suffix)
    elif suffix in ["SEG2", "DAT", "S2", "SG2"]:
        with pytest.raises(ValueError):
            # wrong file descriptor block magic number leads to value error
            seisio.input(file, filetype=suffix)
    else:
        # will work and only output a warning
        sio = seisio.input(file, filetype=suffix)
        assert sio is not None

def test_wrong_input_filetype(tmp_path):
    file = tmp_path / "dummy"
    with open(file, mode="wb") as f:
        f.truncate(8192) # write 8k
    with pytest.raises(ValueError):
        seisio.input(file, filetype="UNKNOWN")

@pytest.mark.parametrize("suffix", ["SGY",
                                    "SEGY",
                                    "SEG-Y",
                                    "SEG_Y",
                                    "SEG2",
                                    "DAT",
                                    "S2",
                                    "SG2",
                                    "SU"])
def test_output(tmp_path, suffix):
    file = tmp_path / "dummy_out"
    if suffix in ["SEG2", "DAT", "S2", "SG2"]:
        with pytest.raises(NotImplementedError):
            sio = seisio.output(file, filetype=suffix)
    elif suffix in ["SGY", "SEGY", "SEG-Y", "SEG_Y"]:
        with pytest.raises(ValueError):
            sio = seisio.output(file, filetype=suffix)
    else:
        with pytest.raises(ValueError):
            sio = seisio.output(file, filetype=suffix)

    if suffix in ["SU"]:
        sio = seisio.output(file, filetype=suffix, ns=10)
        assert sio is not None
        with pytest.raises(ValueError):
            sio = seisio.output(file, filetype=suffix, ns=-10)
        with pytest.raises(TypeError):
            sio = seisio.output(file, filetype=suffix, ns="a")
    elif suffix in ["SGY", "SEGY", "SEG-Y", "SEG_Y", "SU"]:
        sio = seisio.output(file, filetype=suffix, ns=10, vsi=2000)
        assert sio is not None
        with pytest.raises(ValueError):
            sio = seisio.output(file, filetype=suffix, ns=-10, vsi=2000)
        with pytest.raises(ValueError):
            sio = seisio.output(file, filetype=suffix, ns=10, vsi=-2000)
        with pytest.raises(TypeError):
            sio = seisio.output(file, filetype=suffix, ns="a", vsi=2000)
        with pytest.raises(TypeError):
            sio = seisio.output(file, filetype=suffix, ns=10, vsi="a")
    with pytest.raises(ValueError):
        seisio.output(file, filetype="BLA", ns=10)
    with pytest.raises(RuntimeError):
        seisio.output(file, ns=10)
