
import numpy as np
import pytest
import seisio

from pathlib import Path
from seisio import tools
from sys import byteorder


TESTS_DIR = Path(__file__).parent.resolve()


def test_add_mnemonic(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    assert sio is not None

    dataset = sio.read_dataset()

    with pytest.raises(ValueError):
        tools.add_mnemonic(dataset, names=None)
    with pytest.raises(ValueError):
        tools.add_mnemonic(dataset, names="bla", dtypes=None)
    with pytest.raises(TypeError):
        tools.add_mnemonic(dataset, names={"x": 1, "y": 2}, dtypes=np.float32)

    new_dataset = tools.add_mnemonic(dataset, names="tmh", data=True, dtypes=bool)
    assert new_dataset["tmh"][0] == True
    assert new_dataset["tmh"][-1] == True

    new_dataset = tools.add_mnemonic(dataset, names=["tmh1", "tmh2"], data=[99, -11], dtypes=[np.uint32, np.int32])
    assert new_dataset["tmh1"][0] == 99
    assert new_dataset["tmh1"][-1] == 99
    assert new_dataset["tmh2"][0] == -11
    assert new_dataset["tmh2"][-1] == -11

    nt = sio.nt
    data = np.arange(0, nt, 1)
    new_dataset = tools.add_mnemonic(dataset, names="tmh", data=data, dtypes=int)
    assert new_dataset["tmh"][0] == 0
    assert new_dataset["tmh"][-1] == nt-1

    nt = sio.nt
    data = [np.arange(1, nt+1, 1), -11*np.ones((nt,))]
    names = np.array(["tmh1", "tmh2"])
    new_dataset = tools.add_mnemonic(dataset, names=names, data=data, dtypes=[np.uint32, np.int32])
    assert new_dataset["tmh1"][0] == 1
    assert new_dataset["tmh1"][-1] == nt
    assert new_dataset["tmh2"][0] == -11
    assert new_dataset["tmh2"][-1] == -11

    names = np.array(["tmh1", "tmh2"])
    dtypes = "<f4:<f4"
    new_dataset = tools.add_mnemonic(dataset, names=names, dtypes=dtypes)
    assert new_dataset is not None

    names = np.array(["tmh1", "tmh2"])
    dtypes = np.array(["<f4", "<f4"])
    new_dataset = tools.add_mnemonic(dataset, names=names, dtypes=dtypes)
    assert new_dataset is not None

    names = ["tmh1", "tmh2"]
    dtypes = ["<f4"]
    new_dataset = tools.add_mnemonic(dataset, names=names, dtypes=dtypes)
    assert new_dataset is not None

    names = ["tmh1", "tmh2", "tmh3"]
    dtypes = ["<f4", "<f4"]
    with pytest.raises(ValueError):
        tools.add_mnemonic(dataset, names=names, dtypes=dtypes)

    names = ["tmh1", "tmh2"]
    dtypes = ["int", "int"]
    data = 1
    new_dataset = tools.add_mnemonic(dataset, names=names, dtypes=dtypes, data=data)
    assert new_dataset is not None

    names = ["tmh1"]
    data = [np.arange(1, nt+1, 1)]
    dtypes = np.float32
    new_dataset = tools.add_mnemonic(dataset, names=names, dtypes=dtypes, data=data)
    assert new_dataset is not None

    names = ["tmh1", "tmh2"]
    data = [np.arange(1, 3, 1)]
    dtypes = [int, int]
    with pytest.raises(ValueError):
        tools.add_mnemonic(dataset, names=names, dtypes=dtypes, data=data)

    data = [np.arange(1, nt+1, 1), -11*np.ones((4,))]
    names = np.array(["tmh1", "tmh2"])
    dtypes = [int, int]
    with pytest.raises(ValueError):
        tools.add_mnemonic(dataset, names=names, dtypes=dtypes, data=data)

    data = [np.arange(1, nt+1, 1), -11*np.ones((nt,))]
    names = np.array(["tmh1", "tmh2", "tmh3"])
    dtypes = [int, int, int]
    with pytest.raises(ValueError):
        tools.add_mnemonic(dataset, names=names, dtypes=dtypes, data=data)

def test_remove_mnemonic(dummy_segy_file):
    thdefu = TESTS_DIR / "data" / "my_traceheaders.json"
    # fail fast if file was moved or omitted
    if not thdefu.exists():
        pytest.fail(f"User-defined trace header definition file missing at: {thdefu}")
    sio = seisio.input(dummy_segy_file)
    assert sio is not None

    dataset = sio.read_dataset()
    new_dataset = tools.remove_mnemonic(dataset, names="fldr")
    with pytest.raises(ValueError):
        new_dataset["fldr"]

    with pytest.raises(ValueError):
        tools.remove_mnemonic(dataset, names=None)

    new_dataset = tools.remove_mnemonic(dataset, names=["fldr", "tracf", "ens"])
    with pytest.raises(ValueError):
        new_dataset["fldr"]
    with pytest.raises(ValueError):
        new_dataset["tracf"]
    with pytest.raises(ValueError):
        new_dataset["ens"]

    new_dataset = tools.remove_mnemonic(dataset, names={"fldr", "tracf", "ens"})
    with pytest.raises(ValueError):
        new_dataset["fldr"]
    with pytest.raises(ValueError):
        new_dataset["tracf"]
    with pytest.raises(ValueError):
        new_dataset["ens"]

    new_dataset = tools.remove_mnemonic(dataset, allzero=True)
    for n in new_dataset.dtype.names:
        assert np.max(new_dataset[n]) > 0

def test_rename_mnemonic(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    assert sio is not None

    dataset = sio.read_dataset()
    new_dataset = tools.rename_mnemonic(dataset, mapping={"fldr": "shot", "tracf": "tr_in_shot"})
    with pytest.raises(ValueError):
        new_dataset["fldr"]
    with pytest.raises(ValueError):
        new_dataset["tracf"]
    assert np.all(new_dataset["shot"] == dataset["fldr"])
    assert np.all(new_dataset["tr_in_shot"] == dataset["tracf"])

    with pytest.raises(ValueError):
        tools.rename_mnemonic(dataset, mapping=None)

def test_ensemble2cube(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    assert sio is not None

    dataset = sio.read_dataset()
    assert dataset["iline"][0] == 101
    assert dataset["iline"][1] == 101
    assert dataset["iline"][-1] == 103
    assert dataset["iline"][-2] == 103
    assert dataset["xline"][0] == 201
    assert dataset["xline"][1] == 202
    assert dataset["xline"][3] == 204
    assert dataset["xline"][4] == 201

    unstruct = np.lib.recfunctions.structured_to_unstructured(dataset)
    with pytest.raises(ValueError):
        tools.ensemble2cube(unstruct)

    with pytest.raises(ValueError):
        tools.ensemble2cube(dataset, idef=None)

    with pytest.raises(KeyError):
        tools.ensemble2cube(dataset, idef="bla")
    with pytest.raises(KeyError):
        tools.ensemble2cube(dataset, jdef="bla")
    with pytest.raises(KeyError):
        tools.ensemble2cube(dataset, header_trid="bla")

    dummy = np.zeros((0,), dtype=dataset.dtype)
    with pytest.raises(ValueError):
        tools.ensemble2cube(dummy)

    cube = tools.ensemble2cube(dataset, idef="iline", jdef="xline",
                               is_sorted=True, fill_value=0)
    assert cube is not None
    assert cube["data"].shape == (3, 4, 20)

    cube = tools.ensemble2cube(dataset, idef="iline", jdef="xline",
                               is_sorted=False, fill_value=1)
    assert cube is not None
    assert cube["data"].shape == (3, 4, 20)

def test_ensemble2cube_padding(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    assert sio is not None

    dataset = sio.read_dataset()
    cube = tools.ensemble2cube(dataset[0:-1], fill_value=0)
    assert cube is not None

def test_check():
    v = tools._check(None)
    assert isinstance(v, list)
    assert len(v) == 0

def test_check_contiguous():
    buffer = np.array([0, ], dtype=int)
    v = tools._check_if_contiguous(buffer)
    assert v == 0
    buffer = np.array([0, 1, 2, 3], dtype=int)
    v = tools._check_if_contiguous(buffer)
    assert v == 1
    buffer = np.array([0, 2, 9, 23], dtype=int)
    v = tools._check_if_contiguous(buffer)
    assert v == 0

def test_foreign_endianess():
    v = tools._foreign_endian()
    if byteorder == "little":
        assert v == ">"
    else:
        assert v == "<"

def test_native_endianess():
    v = tools._native_endian()
    if byteorder == "little":
         assert v == "<"
    else:
         assert v == ">"

def test_need_swap():
    dtp = np.dtype([("flt", "<f4")])
    v = tools._need_swap(dtp, endian="=")
    if byteorder == "little":
        assert v == False
    else:
        assert v == True

    dtp = np.dtype([("flt", ">f4")])
    v = tools._need_swap(dtp, endian="=")
    if byteorder == "little":
        assert v == True
    else:
        assert v == False

def test_create_dtype():
    names = ["bla"]
    formats = [">f4"]
    v = tools._create_dtype(names, formats, titles=None)
    assert v is not None

def test_create_custom_dtype():
    names = ["bla"]
    formats = [">f4"]
    offsets = [0]
    v = tools._create_custom_dtype(names, formats, offsets, 4, titles=None)
    assert v is not None
