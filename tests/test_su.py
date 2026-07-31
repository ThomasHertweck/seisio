
import json
import numpy as np
import pandas as pd
import pytest
import seisio
import sys

from pathlib import Path
from seisio import tools

def test_open(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    assert sio is not None

    with pytest.raises(ValueError):
        seisio.input(dummy_su_file, endian="99")

    sio = seisio.input(dummy_su_file, invalid_para=42)
    assert sio is not None

    end = sio.endianess
    assert end in [">", "<"]

    sio = seisio.input(dummy_su_file, endian=end)
    assert sio is not None

    v = sio.file
    assert v == Path(dummy_su_file)

    v = sio.fsize
    assert v > 0

    v = sio.dataformat
    assert v == 5

    v = sio.mnemonics
    assert "fldr" in v
    assert "offset" in v
    assert "ns" in v

    v = sio.ns
    assert v == 20

    v = sio.nsamples
    assert v == 20

    v = sio.nt
    assert v == 12

    v = sio.ntraces
    assert v == 12

    v = sio.thsize
    assert v == 240

    v = sio.trsize
    assert v > 240

    v = sio.log_thdef()
    assert v is None

    v = sio.delay
    assert v == 42

    v = sio.vsi
    assert v == 2000

    v = sio.vaxis
    assert v[0]*1000 == 42
    assert v[-1] == pytest.approx(sio.delay*1e-3 + (sio.ns-1)*sio.vsi*1e-6)
    assert len(v) == sio.ns

def test_open_json(dummy_su_file, tmp_path):
    thdef_json = tmp_path / "short.json"
    d = {"tracl": {"byte": 1, "type": "i", "desc": "AAA"},
         "tracr": {"byte": 5, "type": "i", "desc": "BBB"},
         "ns": {"byte": 9, "type": "h", "desc": "CCC"},
         "dt": {"byte": 11, "type": "h", "desc": "DDD"},
         "delrt": {"byte": 13, "type": "h", "desc": "EEE"}}
    thdef_json.write_text(json.dumps(d, indent=4))
    sin = seisio.input(dummy_su_file, thdef=thdef_json)
    assert sin is not None

def test_open_nonexist(tmp_path):
    file_path = tmp_path / "does_not_exist.su"
    with pytest.raises(ValueError):
        seisio.input(file_path)

def test_read_all_headers(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_all_headers()
    assert len(h) == 12
    assert np.max(h["trid"]) == 1
    assert np.min(h["tracr"]) == 99
    assert h["tracl"][0] == 1
    assert h["tracl"][-1] == 12

def test_read_all_headers_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_all_headers(mnemonics=["fldr", "tracf", "ns"])
    assert len(h) == 12
    assert len(h.dtype.names) == 3
    assert h["fldr"][0] == 1
    assert h["fldr"][-1] == 3
    with pytest.raises(ValueError):
        h["trid"]

def test_read_headers(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_headers(0, 5, 11)
    assert len(h) == 3
    assert np.max(h["trid"]) == 1
    assert np.min(h["tracr"]) == 99
    assert h["tracl"][0] == 1
    assert h["tracl"][1] == 6
    assert h["tracl"][2] == 12
    with pytest.raises(ValueError):
        sio.read_headers(12)
    with pytest.raises(ValueError):
        sio.read_headers(-1)

def test_read_headers_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_headers(0, 5, 11, mnemonics=["tracl", "tracf", "ns"])
    assert len(h) == 3
    assert len(h.dtype.names) == 3
    assert h["tracl"][0] == 1
    assert h["tracl"][1] == 6
    assert h["tracl"][2] == 12
    with pytest.raises(ValueError):
        sio.read_headers(12)
    with pytest.raises(ValueError):
        sio.read_headers(-1)
    with pytest.raises(ValueError):
        h["trid"]

def test_read_batch_of_headers(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_batch_of_headers(start=4, nheaders=4)
    assert len(h) == 4
    assert np.max(h["trid"]) == 1
    assert np.min(h["tracr"]) == 99
    assert h["tracl"][0] == 5
    assert h["tracl"][-1] == 8
    with pytest.raises(ValueError):
        sio.read_batch_of_headers(start=-1, nheaders=4)
    with pytest.raises(ValueError):
        sio.read_batch_of_headers(start=12, nheaders=4)
    with pytest.raises(ValueError):
        sio.read_batch_of_headers(start=0, nheaders=13)
    with pytest.raises(ValueError):
        sio.read_batch_of_headers(start=0, nheaders=-4)

def test_read_batch_of_headers_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_batch_of_headers(start=4, nheaders=4, mnemonics=["tracl", "tracf", "ns"])
    assert len(h) == 4
    assert len(h.dtype.names) == 3
    assert h["tracl"][0] == 5
    assert h["tracl"][-1] == 8
    with pytest.raises(ValueError):
        sio.read_batch_of_headers(start=-1, nheaders=4, mnemonics=["tracl", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_batch_of_headers(start=12, nheaders=4, mnemonics=["tracl", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_batch_of_headers(start=0, nheaders=13, mnemonics=["tracl", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_batch_of_headers(start=0, nheaders=-4, mnemonics=["tracl", "tracf", "ns"])

def test_read_multibatch_of_headers(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_multibatch_of_headers(start=0, block=2, count=3, stride=4)
    assert len(h) == 6
    assert np.max(h["trid"]) == 1
    assert np.min(h["tracr"]) == 99
    assert h["fldr"][0] == 1
    assert h["fldr"][2] == 2
    assert h["fldr"][4] == 3
    assert h["tracl"][-1] == 10
    with pytest.raises(ValueError):
        sio.read_multibatch_of_headers(start=-1, block=2, count=3, stride=4)
    with pytest.raises(ValueError):
        sio.read_multibatch_of_headers(start=12, block=2, count=3, stride=4)
    with pytest.raises(ValueError):
        sio.read_multibatch_of_headers(start=0, block=6, count=3, stride=4)
    with pytest.raises(ValueError):
        sio.read_multibatch_of_headers(start=0, block=2, count=6, stride=4)
    with pytest.raises(ValueError):
        sio.read_multibatch_of_headers(start=0, block=2, count=3, stride=12)
    with pytest.raises(ValueError):
        sio.read_multibatch_of_headers(start=0, block=0, count=1, stride=1)

def test_read_multibatch_of_headers_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_multibatch_of_headers(start=0, block=2, count=3, stride=4,
                                       mnemonics=["fldr", "tracf", "ns"])
    assert len(h) == 6
    assert len(h.dtype.names) == 3
    assert h["fldr"][0] == 1
    assert h["fldr"][2] == 2
    assert h["fldr"][4] == 3
    assert h["tracf"][-1] == 2
    with pytest.raises(ValueError):
        h["trid"]
    with pytest.raises(ValueError):
        sio.read_multibatch_of_headers(start=-1, block=2, count=3, stride=4,
                                       mnemonics=["fldr", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_multibatch_of_headers(start=12, block=2, count=3, stride=4,
                                       mnemonics=["fldr", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_multibatch_of_headers(start=0, block=6, count=3, stride=4,
                                       mnemonics=["fldr", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_multibatch_of_headers(start=0, block=2, count=6, stride=4,
                                       mnemonics=["fldr", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_multibatch_of_headers(start=0, block=2, count=3, stride=12,
                                       mnemonics=["fldr", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_multibatch_of_headers(start=0, block=0, count=1, stride=1,
                                       mnemonics=["fldr", "tracf", "ns"])

def test_batches_of_headers(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    for i, h in enumerate(sio.batches_of_headers(batch_size=4)):
        assert len(h) == 4
        assert h["fldr"][0] == 1+i
        assert h["tracl"][0] == 1+i*4

def test_batches_of_headers_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    for i, h in enumerate(sio.batches_of_headers(batch_size=4, mnemonics=["fldr", "tracl", "ns"])):
        assert len(h) == 4
        assert len(h.dtype.names) == 3
        assert h["fldr"][0] == 1+i
        assert h["tracl"][0] == 1+i*4
        with pytest.raises(ValueError):
            h["trid"]

def test_headers(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    for i, h in enumerate(sio.headers()):
         assert h["tracl"][0] == 1+i

def test_headers_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    for i, h in enumerate(sio.headers(mnemonics=["fldr", "tracl", "ns"])):
        assert len(h.dtype.names) == 3
        assert h["tracl"][0] == 1+i
        with pytest.raises(ValueError):
            h["trid"]

def test_read_all_traces(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_all_traces()
    assert len(h) == 12
    assert np.max(h["trid"]) == 1
    assert np.min(h["tracr"]) == 99
    assert h["tracl"][0] == 1
    assert h["tracl"][-1] == 12
    hist = []
    h = sio.read_all_traces(history=hist)
    assert len(hist) == 1

def test_read_all_traces_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_all_traces(mnemonics=["fldr", "tracf", "ns"])
    assert len(h) == 12
    assert len(h.dtype.names) == 4
    assert h["fldr"][0] == 1
    assert h["fldr"][-1] == 3
    with pytest.raises(ValueError):
        h["trid"]
    hist = []
    h = sio.read_all_traces(mnemonics=["fldr", "tracf", "ns"], history=hist)
    assert len(hist) == 1

def test_read_dataset(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_dataset()
    assert len(h) == 12
    assert np.max(h["trid"]) == 1
    assert np.min(h["tracr"]) == 99
    assert h["tracl"][0] == 1
    assert h["tracl"][-1] == 12
    hist = []
    h = sio.read_dataset(history=hist)
    assert len(hist) == 1

def test_read_dataset_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_dataset(mnemonics=["fldr", "tracf", "ns"])
    assert len(h) == 12
    assert len(h.dtype.names) == 4
    assert h["fldr"][0] == 1
    assert h["fldr"][-1] == 3
    with pytest.raises(ValueError):
        h["trid"]
    hist = []
    h = sio.read_dataset(history=hist, mnemonics=["fldr", "tracf", "ns"])
    assert len(hist) == 1

def test_read_traces(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_traces(0, 5, 11)
    assert len(h) == 3
    assert np.max(h["trid"]) == 1
    assert np.min(h["tracr"]) == 99
    assert h["tracl"][0] == 1
    assert h["tracl"][1] == 6
    assert h["tracl"][2] == 12
    with pytest.raises(ValueError):
        sio.read_traces(12)
    with pytest.raises(ValueError):
        sio.read_traces(-1)
    hist = []
    h = sio.read_traces(0, 5, 11, history=hist)
    assert len(hist) == 1

def test_read_traces_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_traces(0, 5, 11, mnemonics=["tracl", "tracf", "ns"])
    assert len(h) == 3
    assert len(h.dtype.names) == 4
    assert h["tracl"][0] == 1
    assert h["tracl"][1] == 6
    assert h["tracl"][2] == 12
    with pytest.raises(ValueError):
        sio.read_traces(12)
    with pytest.raises(ValueError):
        sio.read_traces(-1)
    with pytest.raises(ValueError):
        h["trid"]
    hist = []
    h = sio.read_traces(0, 5, 11, mnemonics=["tracl", "tracf", "ns"], history=hist)
    assert len(hist) == 1

def test_read_batch_of_traces(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_batch_of_traces(start=4, ntraces=4)
    assert len(h) == 4
    assert np.max(h["trid"]) == 1
    assert np.min(h["tracr"]) == 99
    assert h["tracl"][0] == 5
    assert h["tracl"][-1] == 8
    with pytest.raises(ValueError):
        sio.read_batch_of_traces(start=-1, ntraces=4)
    with pytest.raises(ValueError):
        sio.read_batch_of_traces(start=12, ntraces=4)
    with pytest.raises(ValueError):
        sio.read_batch_of_traces(start=0, ntraces=13)
    with pytest.raises(ValueError):
        sio.read_batch_of_traces(start=0, ntraces=-4)
    hist = []
    h = sio.read_batch_of_traces(start=4, ntraces=4, history=hist)
    assert len(hist) == 1

def test_read_batch_of_traces_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_batch_of_traces(start=4, ntraces=4, mnemonics=["tracl", "tracf", "ns"])
    assert len(h) == 4
    assert len(h.dtype.names) == 4
    assert h["tracl"][0] == 5
    assert h["tracl"][-1] == 8
    with pytest.raises(ValueError):
        sio.read_batch_of_traces(start=-1, ntraces=4, mnemonics=["tracl", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_batch_of_traces(start=12, ntraces=4, mnemonics=["tracl", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_batch_of_traces(start=0, ntraces=13, mnemonics=["tracl", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_batch_of_traces(start=0, ntraces=-4, mnemonics=["tracl", "tracf", "ns"])
    hist = []
    h = sio.read_batch_of_traces(start=4, ntraces=4, mnemonics=["tracl", "tracf", "ns"], history=hist)
    assert len(hist) == 1

def test_read_multibatch_of_traces(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_multibatch_of_traces(start=0, block=2, count=3, stride=4)
    assert len(h) == 6
    assert np.max(h["trid"]) == 1
    assert np.min(h["tracr"]) == 99
    assert h["fldr"][0] == 1
    assert h["fldr"][2] == 2
    assert h["fldr"][4] == 3
    assert h["tracl"][-1] == 10
    with pytest.raises(ValueError):
        sio.read_multibatch_of_traces(start=-1, block=2, count=3, stride=4)
    with pytest.raises(ValueError):
        sio.read_multibatch_of_traces(start=12, block=2, count=3, stride=4)
    with pytest.raises(ValueError):
        sio.read_multibatch_of_traces(start=0, block=6, count=3, stride=4)
    with pytest.raises(ValueError):
        sio.read_multibatch_of_traces(start=0, block=2, count=6, stride=4)
    with pytest.raises(ValueError):
        sio.read_multibatch_of_traces(start=0, block=2, count=3, stride=12)
    with pytest.raises(ValueError):
        sio.read_multibatch_of_traces(start=0, block=0, count=1, stride=1)
    hist = []
    h = sio.read_multibatch_of_traces(start=0, block=2, count=3, stride=4, history=hist)
    assert len(hist) == 1

def test_read_multibatch_of_traces_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_multibatch_of_traces(start=0, block=2, count=3, stride=4,
                                       mnemonics=["fldr", "tracf", "ns"])
    assert len(h) == 6
    assert len(h.dtype.names) == 4
    assert h["fldr"][0] == 1
    assert h["fldr"][2] == 2
    assert h["fldr"][4] == 3
    assert h["tracf"][-1] == 2
    with pytest.raises(ValueError):
        h["trid"]
    with pytest.raises(ValueError):
        sio.read_multibatch_of_traces(start=-1, block=2, count=3, stride=4,
                                      mnemonics=["fldr", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_multibatch_of_traces(start=12, block=2, count=3, stride=4,
                                      mnemonics=["fldr", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_multibatch_of_traces(start=0, block=6, count=3, stride=4,
                                      mnemonics=["fldr", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_multibatch_of_traces(start=0, block=2, count=6, stride=4,
                                      mnemonics=["fldr", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_multibatch_of_traces(start=0, block=2, count=3, stride=12,
                                      mnemonics=["fldr", "tracf", "ns"])
    with pytest.raises(ValueError):
        sio.read_multibatch_of_traces(start=0, block=0, count=1, stride=1,
                                      mnemonics=["fldr", "tracf", "ns"])
    hist = []
    h = sio.read_multibatch_of_traces(start=0, block=2, count=3, stride=4,
                                       mnemonics=["fldr", "tracf", "ns"], history=hist)
    assert len(hist) == 1

def test_batches(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    for i, h in enumerate(sio.batches(batch_size=4)):
        assert len(h) == 4
        assert h["fldr"][0] == 1+i
        assert h["tracl"][0] == 1+i*4
    hist = []
    for i, h in enumerate(sio.batches(batch_size=4, history=hist)):
        assert len(hist) == i+1

def test_batches_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    for i, h in enumerate(sio.batches(batch_size=4, mnemonics=["fldr", "tracl", "ns"])):
        assert len(h) == 4
        assert len(h.dtype.names) == 4
        assert h["fldr"][0] == 1+i
        assert h["tracl"][0] == 1+i*4
        with pytest.raises(ValueError):
            h["trid"]
    hist = []
    for i, h in enumerate(sio.batches(batch_size=4, mnemonics=["fldr", "tracl", "ns"],
                                      history=hist)):
        assert len(hist) == i+1

def test_traces(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    for i, h in enumerate(sio.traces()):
         assert h["tracl"][0] == 1+i
    hist = []
    for i, h in enumerate(sio.traces(history=hist)):
         assert len(hist) == i+1

def test_traces_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    for i, h in enumerate(sio.traces(mnemonics=["fldr", "tracl", "ns"])):
        assert len(h.dtype.names) == 4
        assert h["tracl"][0] == 1+i
        with pytest.raises(ValueError):
            h["trid"]
    hist = []
    for i, h in enumerate(sio.traces(mnemonics=["fldr", "tracl", "ns"], history=hist)):
        assert len(hist) == i+1

def test_create_index(dummy_su_file):
    sio = seisio.input(dummy_su_file)

    sio.create_index(group_by="fldr", sort_by="tracf")
    assert sio.ne == 3
    assert sio.nensembles == 3
    assert sio.maxnte == 4
    assert len(sio.nte) == 3
    assert sio.nte[0] == 4
    assert sio.nte[-1] == 4
    assert len(sio.ensemble_keys) == 3

    sio.create_index(group_by="fldr", sort_by="tracf", group_order="<", sort_order="<")
    assert sio.ne == 3
    assert sio.nensembles == 3
    assert sio.maxnte == 4
    assert len(sio.nte) == 3
    assert sio.nte[0] == 4
    assert sio.nte[-1] == 4
    assert len(sio.ensemble_keys) == 3

    h = sio.read_all_headers()
    sio.create_index(group_by="cdp", sort_by="offset", headers=h)
    assert sio.ne == 3
    assert sio.nensembles == 3
    assert sio.maxnte == 4
    assert len(sio.nte) == 3
    assert sio.nte[0] == 4
    assert sio.nte[-1] == 4
    assert len(sio.ensemble_keys) == 3

    def filt_func(x): return (x["fldr"] < 3)
    sio.create_index(group_by="fldr", sort_by="tracf", filt=filt_func)
    assert sio.ne == 2
    assert sio.nensembles == 2
    assert sio.maxnte == 4
    assert len(sio.nte) == 2
    assert sio.nte[0] == 4
    assert sio.nte[-1] == 4
    assert len(sio.ensemble_keys) == 2

    sio.create_index(group_by=["fldr", "tracf"], sort_by=["tracr"])
    assert sio.ne == 12
    assert sio.nensembles == 12
    assert sio.maxnte == 1
    assert len(sio.nte) == 12
    assert sio.nte[0] == 1
    assert sio.nte[-1] == 1
    assert len(sio.ensemble_keys) == 12

def test_read_ensemble_default(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    sio.create_index(group_by="fldr", sort_by="tracf")
    ens = sio.read_ensemble(sio.ensemble_keys[-1])
    assert ens["fldr"][0] == 3
    assert ens["tracf"][-1] == 4
    data = ens["data"]
    assert data.shape == (4, 20)
    assert data[0, 0] == 8000
    assert data[3, 19] == 11019
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], history=hist)
    assert len(hist) == 1

def test_read_ensemble_default_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    sio.create_index(group_by="fldr", sort_by="tracf")
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics=["fldr", "tracf"])
    assert ens["fldr"][0] == 3
    assert ens["tracf"][-1] == 4
    with pytest.raises(ValueError):
        ens["trid"]
    data = ens["data"]
    assert data.shape == (4, 20)
    assert data[0, 0] == 8000
    assert data[3, 19] == 11019
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics=["fldr", "tracf"], history=hist)
    assert len(hist) == 1

def test_read_ensemble_order(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    sio.create_index(group_by="fldr", sort_by="tracf", group_order="<", sort_order="<")
    ens = sio.read_ensemble(sio.ensemble_keys[-1])
    assert ens["fldr"][0] == 1
    assert ens["tracf"][-1] == 1
    data = ens["data"]
    assert data.shape == (4, 20)
    assert data[0, 0] == 3000
    assert data[3, 19] == 19
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], history=hist)
    assert len(hist) == 1

def test_read_ensemble_order_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    sio.create_index(group_by="fldr", sort_by="tracf", group_order="<", sort_order="<")
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics=["fldr", "tracf"])
    assert ens["fldr"][0] == 1
    assert ens["tracf"][-1] == 1
    with pytest.raises(ValueError):
        ens["trid"]
    data = ens["data"]
    assert data.shape == (4, 20)
    assert data[0, 0] == 3000
    assert data[3, 19] == 19
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics=["fldr", "tracf"], history=hist)
    assert len(hist) == 1

def test_read_ensemble_filt(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    def filt_func(x): return (x["fldr"] < 3)
    sio.create_index(group_by="fldr", sort_by="tracf", filt=filt_func)
    ens = sio.read_ensemble(sio.ensemble_keys[-1])
    assert ens["fldr"][0] == 2
    assert ens["tracf"][-1] == 4
    data = ens["data"]
    assert data.shape == (4, 20)
    assert data[0, 0] == 4000
    assert data[3, 19] == 7019
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], history=hist)
    assert len(hist) == 1

def test_read_ensemble_filt_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    def filt_func(x): return (x["fldr"] < 3)
    sio.create_index(group_by="fldr", sort_by="tracf", filt=filt_func)
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics=["fldr", "tracf"])
    assert ens["fldr"][0] == 2
    assert ens["tracf"][-1] == 4
    with pytest.raises(ValueError):
        ens["trid"]
    data = ens["data"]
    assert data.shape == (4, 20)
    assert data[0, 0] == 4000
    assert data[3, 19] == 7019
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics=["fldr", "tracf"], history=hist)
    assert len(hist) == 1

def test_read_ensemble_multi(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    sio.create_index(group_by=["fldr", "cdp"], sort_by=["tracf"])
    ens = sio.read_ensemble(sio.ensemble_keys[-1])
    assert ens["fldr"][0] == 3
    assert ens["tracf"][-1] == 4
    data = ens["data"]
    assert data.shape == (4, 20)
    assert data[0, 19] == 8019
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], history=hist)
    assert len(hist) == 1

def test_read_ensemble_multi_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    sio.create_index(group_by=["fldr", "cdp"], sort_by=["tracf"])
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics=["fldr", "tracf"])
    assert ens["fldr"][0] == 3
    assert ens["tracf"][-1] == 4
    with pytest.raises(ValueError):
        ens["trid"]
    data = ens["data"]
    assert data.shape == (4, 20)
    assert data[0, 19] == 8019
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics=["fldr", "tracf"], history=hist)
    assert len(hist) == 1

def test_read_ensembles(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    hist = []
    sio.create_index(group_by="fldr", sort_by="tracf")
    for i, ens in enumerate(sio.ensembles(history=hist)):
        assert ens["fldr"][0] == 1+i
        assert ens["tracl"][0] == 1+i*4
        data = ens["data"]
        assert data.shape == (4, 20)
        assert data[0, 19] == i*4000+19
        assert len(hist) == i+1

def test_read_ensembles_mnemonics(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    sio.create_index(group_by="fldr", sort_by="tracf")
    hist = []
    for i, ens in enumerate(sio.ensembles(mnemonics=["fldr", "tracl"], history=hist)):
        assert ens["fldr"][0] == 1+i
        assert ens["tracl"][0] == 1+i*4
        with pytest.raises(ValueError):
            ens["trid"]
        data = ens["data"]
        assert data.shape == (4, 20)
        assert data[0, 19] == i*4000+19
        assert len(hist) == i+1

def test_thstat_default(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    thstat1 = sio.thstat()
    assert thstat1 is not None
    thstat2 = sio.thstat(ntmax=10)
    assert thstat2 is not None
    with pytest.raises(ValueError):
        sio.thstat(ntmax=99)
    thstat4 = sio.log_thstat(thstat=thstat1)
    assert thstat4 is not None
    thstat5 = sio.log_thstat(thstat=thstat2)
    assert thstat5 is not None

def test_thstat_memory(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    h = sio.read_all_headers()
    thstat1 = sio.thstat(headers=h)
    assert thstat1 is not None
    thstat2 = sio.thstat(headers=h, ntmax=10)
    assert thstat2 is not None
    thstat3 = sio.thstat(headers=h, ntmax=99)
    assert thstat3 is not None

def test_write_little(tmp_path, dummy_su_file):
    file = tmp_path / "dummy_out_little.su"

    sin = seisio.input(dummy_su_file)
    data_ref = sin.read_dataset()

    with pytest.raises(ValueError):
        seisio.output(file, ns=sin.ns, endian="XY")

    sol = seisio.output(file, ns=sin.ns, unknown_para=42)

    head_l = sol.headers_template(nt=sin.nt)
    assert len(head_l) == sin.nt
    head_l_pd = sol.headers_template(nt=sin.nt, pandas=True)
    assert len(head_l_pd) == sin.nt
    trac_l = sol.traces_template(nt=sin.nt, headers_only=False)
    assert len(trac_l) == sin.nt
    assert trac_l["data"].shape == (sin.nt, sin.ns)
    trac_l_hd = sol.traces_template(nt=sin.nt, headers_only=True)
    assert np.all(trac_l_hd == head_l)

    nt = sol.write_traces(traces=sin.read_dataset())
    assert nt == sin.nt

    nt += sol.write_traces(data=sin.read_dataset()["data"], headers=sin.read_all_headers())
    assert nt == 2*sin.nt

    nt += sol.write_traces(traces=sin.read_dataset(), remap={"fldr": "ep"})
    assert nt == 3*sin.nt

    sil = seisio.input(file)
    assert sil.nt == 3*sin.nt
    assert sil.ns == 20

    data_rdb = sil.read_dataset()
    assert np.max(data_ref["ep"]) == 0
    assert data_rdb["ep"][-1] == 3
    assert np.all(data_rdb["data"][0:sin.nt, :] == data_ref["data"][:, :])
    assert np.all(data_rdb["ep"][2*sin.nt:] == data_ref["fldr"])
    assert np.all(data_rdb["fldr"][sin.nt:2*sin.nt] == data_ref["fldr"])

def test_write_big(tmp_path, dummy_su_file):
    file = tmp_path / "dummy_out_big.su"

    sin = seisio.input(dummy_su_file)
    data_ref = sin.read_dataset()

    sol = seisio.output(file, ns=sin.ns, endian=">")

    head_l = sol.headers_template(nt=sin.nt)
    assert len(head_l) == sin.nt
    head_l_pd = sol.headers_template(nt=sin.nt, pandas=True)
    assert len(head_l_pd) == sin.nt
    trac_l = sol.traces_template(nt=sin.nt, headers_only=False)
    assert len(trac_l) == sin.nt
    assert trac_l["data"].shape == (sin.nt, sin.ns)
    trac_l_hd = sol.traces_template(nt=sin.nt, headers_only=True)
    assert np.all(trac_l_hd == head_l)

    nt = sol.write_traces(traces=sin.read_dataset())
    assert nt == sin.nt

    nt += sol.write_traces(data=sin.read_dataset()["data"], headers=sin.read_all_headers())
    assert nt == 2*sin.nt

    nt += sol.write_traces(traces=sin.read_dataset(), remap={"fldr": "ep"})
    assert nt == 3*sin.nt

    sil = seisio.input(file)
    assert sil.nt == 3*sin.nt
    assert sil.ns == 20

    data_rdb = sil.read_dataset()
    assert np.max(data_ref["ep"]) == 0
    assert data_rdb["ep"][-1] == 3
    assert np.all(data_rdb["data"][0:sin.nt, :] == data_ref["data"][:, :])
    assert np.all(data_rdb["ep"][2*sin.nt:] == data_ref["fldr"])
    assert np.all(data_rdb["fldr"][sin.nt:2*sin.nt] == data_ref["fldr"])

def test_log_thstat(dummy_su_file):
    sio = seisio.input(dummy_su_file)
    dataset = sio.read_dataset()
    thstat1 = seisio.log_thstat(dataset, zero=False)
    assert thstat1 is not None
    thstat2= seisio.log_thstat(dataset, zero=True)
    assert thstat2 is not None

def test_log_thstat_patch(dummy_su_file, monkeypatch):
    monkeypatch.delitem(sys.modules, "tabulate", raising=False)
    monkeypatch.setitem(sys.modules, "tabulate", None)
    sio = seisio.input(dummy_su_file)
    dataset = sio.read_dataset()
    thstat1 = seisio.log_thstat(dataset, zero=False)
    assert thstat1 is not None
    thstat2= seisio.log_thstat(dataset, zero=True)
    assert thstat2 is not None

def test_undecided_endianess(dummy_special_su_file):
    sio = seisio.input(dummy_special_su_file)
    assert sio is not None
    assert sio.endianess == "="

def test_write(tmp_path, dummy_su_file):
    file_path = tmp_path / "dummy_out_little.su"

    sin = seisio.input(dummy_su_file)
    assert sin is not None
    dataset = sin.read_dataset()
    ns = sin.ns

    with pytest.raises(ValueError):
        seisio.output(file_path, ns=ns, mode="x")

    sol = seisio.output(file_path, ns=ns, mode="a")
    assert sol is not None

    with open(file_path, "wb") as f:
        f.write("000".encode("ascii"))

    sol = seisio.output(file_path, ns=ns, mode="a")
    assert sol is not None

    with pytest.raises(ValueError):
        sol._headers_transfer(None)

    with pytest.raises(ValueError):
        sol.write_traces(data=None, headers=None, traces=None)
    with pytest.raises(ValueError):
        sol.write_traces(data=dataset["data"], headers=None, traces=dataset)
    with pytest.raises(ValueError):
        sol.write_traces(data=None, headers=dataset["fldr"], traces=dataset)
    with pytest.raises(ValueError):
        sol.write_traces(data=None, headers=dataset["fldr"])
    with pytest.raises(ValueError):
        sol.write_traces(data=dataset["data"], headers=None)

    sol._head_written = False
    with pytest.raises(RuntimeError):
        sol.write_traces(traces=dataset)
    sol._head_written = True

    sol._tail_written = True
    with pytest.raises(RuntimeError):
        sol.write_traces(traces=dataset)
    sol._tail_written = False

    dummy = np.zeros((0,), dtype=dataset.dtype)
    with pytest.raises(ValueError):
        sol.write_traces(traces=dummy)

    new_dataset = tools.remove_mnemonic(dataset, names=["data"])
    with pytest.raises(ValueError):
        sol.write_traces(traces=new_dataset)

    sol._dp.ns = 999
    with pytest.raises(ValueError):
        sol.write_traces(data=dataset["data"], headers=None)
    sol._dp.ns = ns

    sol = seisio.output(file_path, ns=ns)
    assert sol is not None

    data = dataset["data"].copy()
    headers = new_dataset
    df = pd.DataFrame(headers)
    with pytest.raises(TypeError):
        sol.write_traces(data=data, headers=99)
    nt = sol.write_traces(data=data, headers=df)
    assert nt == sin.nt

    data2 = np.zeros_like(data, dtype=int)
    nt += sol.write_traces(data=data2, headers=headers)
    assert nt == 2*sin.nt
