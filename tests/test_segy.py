
import json
import numpy as np
import pandas as pd
import pytest
import seisio
import sys

from pathlib import Path
from seisio import tools

TESTS_DIR = Path(__file__).parent.resolve()


def test_open(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    assert sio is not None

    with pytest.raises(ValueError):
        seisio.input(dummy_segy_file, endian="99")

    sio = seisio.input(dummy_segy_file, invalid_para=42)
    assert sio is not None

    end = sio.endianess
    assert end in [">", "<"]

    sio = seisio.input(dummy_segy_file, endian=end)
    assert sio is not None

    v = sio.file
    assert v == Path(dummy_segy_file)

    v = sio.fsize
    assert v > 0

    v = sio.dataformat
    assert v > 0 and v <= 16

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

    thsize = sio.thsize
    assert thsize >= 240

    v = sio.trsize
    assert v > thsize

    v = sio.delay
    assert v == 10

    v = sio.coord_scaler
    assert v == 1
    
    v = sio.elev_scaler
    assert v == -1

    v = sio.vsi
    assert v == 2000

    v = sio.vaxis
    assert v[0]*1000 == 10
    assert v[-1] == pytest.approx(sio.delay*1e-3 + (sio.ns-1)*sio.vsi*1e-6)
    assert len(v) == sio.ns

    v = sio.log_thdef()
    assert v is None

    v = sio.binhead
    assert isinstance(v, np.ndarray)

    v = sio.nthuser
    assert v >= 0

    v = sio.ntxtrail
    assert v >= 0

    v = sio.ntxtrec
    assert v >= 0

    v = sio.records
    assert v is not None

    v = sio.thext1
    assert isinstance(v, bool)

    v = sio.trailers
    assert v is not None

    v = sio.txthead
    assert len(v) == 40
    
    v = sio.trdtype
    assert isinstance(v, np.dtype)

    v = sio.thdtype
    assert isinstance(v, np.dtype)

    with pytest.raises(NotImplementedError):
        seisio.input(dummy_segy_file, fixed=False)

    with pytest.raises(TypeError):
        seisio.input(dummy_segy_file, thext1=42)

    with pytest.raises(ValueError):
        seisio.input(dummy_segy_file, ntxtrec=-10)

    with pytest.raises(ValueError):
        seisio.input(dummy_segy_file, ntxtrail=-10)

    with pytest.raises(ValueError):
        seisio.input(dummy_segy_file, nthuser=-10)

    with pytest.raises(ValueError):
        seisio.input(dummy_segy_file, format=15)

    sio = seisio.input(dummy_segy_file, fixed=True)
    assert sio is not None

    v = sio.ensemble_keys
    assert len(v) == 0

    v = sio.ne
    assert v == 0

    v = sio.nte
    assert len(v) == 1
    assert v[0] == 0

    with pytest.raises(RuntimeError):
        sio.read_ensemble(0)

def test_open_special(dummy_segy_file_special):
    thdefu = TESTS_DIR / "data" / "my_traceheaders.json"
    # fail fast if file was moved or omitted
    if not thdefu.exists():
        pytest.fail(f"User-defined trace header definition file missing at: {thdefu}")
    sio = seisio.input(dummy_segy_file_special, thdefu=thdefu)
    assert sio is not None

    sio = seisio.input(dummy_segy_file_special, thdefu=thdefu, ntxtrail=1)
    assert sio is not None

    sio = seisio.input(dummy_segy_file_special, txext1=True, nthuser=1,
                       thdefu=thdefu, ntxtrail=1)
    assert sio is not None

    sio = seisio.input(dummy_segy_file_special, thdefu=thdefu, ntxtrail=1)
    assert sio is not None

def test_open_json(dummy_segy_file, tmp_path):
    thdef_json = tmp_path / "short.json"
    d = {"tracl": {"byte": 1, "type": "i", "desc": "AAA"},
         "tracr": {"byte": 5, "type": "i", "desc": "BBB"},
         "ns": {"byte": 9, "type": "h", "desc": "CCC"},
         "dt": {"byte": 11, "type": "h", "desc": "DDD"},
         "delrt": {"byte": 13, "type": "h", "desc": "EEE"},
         "scalel": {"byte": 15, "type": "h", "desc": "FFF"},
         "scalco": {"byte": 17, "type": "h", "desc": "GGG"}}
    thdef_json.write_text(json.dumps(d, indent=4))
    sin = seisio.input(dummy_segy_file, thdef=thdef_json)
    assert sin is not None

def test_get_binhead(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    bh = sio.get_binhead()
    assert bh["ns"] == 20
    assert bh["dt"] == 2000
    assert bh["format"] > 0 and bh["format"] <= 16
    if sio.dataformat not in [8, 16]:
        assert bh["fixed"] == 1

def test_get_records(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    rec = sio.get_records()
    assert rec is not None
    for i in range(len(rec)):
        assert "EXTENDED HEADER" in rec[i]

def test_get_trailers(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    trl = sio.get_trailers(decode=True)
    assert trl is not None
    for i in range(len(trl)):
        assert "REV2 TRAILER" in trl[i]

def test_get_trailers_nodecode(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    trl = sio.get_trailers(decode=False)
    assert trl is not None

def test_get_txthead(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    th = sio.get_txthead()
    assert th is not None
    assert "DUMMY FILE HEADER LINE 11" in th[10]

def test_log_bhdef(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    v = sio.log_bhdef()
    assert v is None

def test_log_binhead(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    v = sio.log_binhead()
    assert v is not None
    assert v["ns"] == 20

def test_log_binhead_zero(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    v = sio.log_binhead(zero=True)
    assert v is not None
    assert v["ns"] == 20

def test_log_txthead(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    th = sio.log_txthead()
    assert th is not None
    assert "DUMMY FILE HEADER LINE 11" in th[10]

def test_read_all_headers(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    h = sio.read_all_headers()
    assert len(h) == 12
    assert np.max(h["trid"]) == 1
    assert np.min(h["tracr"]) == 99
    assert h["tracl"][0] == 1
    assert h["tracl"][-1] == 12

def test_read_all_headers_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    h = sio.read_all_headers(mnemonics=["fldr", "tracf", "ns"])
    assert len(h) == 12
    assert len(h.dtype.names) == 3
    assert h["fldr"][0] == 1
    assert h["fldr"][-1] == 3
    with pytest.raises(ValueError):
        h["trid"]

def test_read_headers(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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
    with pytest.raises(ValueError):
        sio.read_headers()

def test_read_headers_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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

def test_read_batch_of_headers(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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

def test_read_batch_of_headers_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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

def test_read_multibatch_of_headers(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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
    with pytest.raises(ValueError):
        sio.read_multibatch_of_headers(count=None, stride=None, block=None)

def test_read_multibatch_of_headers_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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

def test_batches_of_headers(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    for i, h in enumerate(sio.batches_of_headers(batch_size=4)):
        assert len(h) == 4
        assert h["fldr"][0] == 1+i
        assert h["tracl"][0] == 1+i*4
    with pytest.raises(ValueError):
        for h in sio.batches_of_headers(batch_size=-1):
            pass

def test_batches_of_headers_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    for i, h in enumerate(sio.batches_of_headers(batch_size=4, mnemonics=["fldr", "tracl", "ns"])):
        assert len(h) == 4
        assert len(h.dtype.names) == 3
        assert h["fldr"][0] == 1+i
        assert h["tracl"][0] == 1+i*4
        with pytest.raises(ValueError):
            h["trid"]

def test_headers(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    for i, h in enumerate(sio.headers()):
         assert h["tracl"][0] == 1+i

def test_headers_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    for i, h in enumerate(sio.headers(mnemonics=["fldr", "tracl", "ns"])):
        assert len(h.dtype.names) == 3
        assert h["tracl"][0] == 1+i
        with pytest.raises(ValueError):
            h["trid"]

def test_read_all_traces(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    h = sio.read_all_traces()
    assert len(h) == 12
    assert np.max(h["trid"]) == 1
    assert np.min(h["tracr"]) == 99
    assert h["tracl"][0] == 1
    assert h["tracl"][-1] == 12
    hist = []
    h = sio.read_all_traces(history=hist)
    assert len(hist) == 1

def test_read_all_traces_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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

def test_read_dataset(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    h = sio.read_dataset()
    assert len(h) == 12
    assert np.max(h["trid"]) == 1
    assert np.min(h["tracr"]) == 99
    assert h["tracl"][0] == 1
    assert h["tracl"][-1] == 12
    hist = []
    h = sio.read_dataset(history=hist)
    assert len(hist) == 1

def test_read_dataset_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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

def test_read_traces(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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
    with pytest.raises(ValueError):
        sio.read_traces()

def test_read_traces_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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

def test_read_batch_of_traces(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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
    with pytest.raises(TypeError):
        sio.read_batch_of_traces(count=None, block=None, stride=None)

def test_read_batch_of_traces_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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

def test_read_multibatch_of_traces(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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
    with pytest.raises(ValueError):
        sio.read_multibatch_of_traces(count=None, block=None, stride=None)

def test_read_multibatch_of_traces_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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

def test_batches(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    for i, h in enumerate(sio.batches(batch_size=4)):
        assert len(h) == 4
        assert h["fldr"][0] == 1+i
        assert h["tracl"][0] == 1+i*4
    hist = []
    for i, h in enumerate(sio.batches(batch_size=4, history=hist)):
        assert len(hist) == i+1
    with pytest.raises(ValueError):
        for h in sio.batches(batch_size=-1):
            pass

def test_batches_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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

def test_traces(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    for i, h in enumerate(sio.traces()):
         assert h["tracl"][0] == 1+i
    hist = []
    for i, h in enumerate(sio.traces(history=hist)):
         assert len(hist) == i+1

def test_traces_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    for i, h in enumerate(sio.traces(mnemonics=["fldr", "tracl", "ns"])):
        assert len(h.dtype.names) == 4
        assert h["tracl"][0] == 1+i
        with pytest.raises(ValueError):
            h["trid"]
    hist = []
    for i, h in enumerate(sio.traces(mnemonics=["fldr", "tracl", "ns"], history=hist)):
        assert len(hist) == i+1

def test_create_index(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)

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

    with pytest.raises(ValueError):
        sio.create_index(group_by=["bla", "tracf"], sort_by=["tracr"])
    with pytest.raises(ValueError):
        sio.create_index(group_by=["fldr", "tracf"], sort_by=["bla"])

def test_read_ensemble_default(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    sio.create_index(group_by="fldr", sort_by="tracf")
    ens = sio.read_ensemble(sio.ensemble_keys[-1])
    assert ens["fldr"][0] == 3
    assert ens["tracf"][-1] == 4
    data = ens["data"]
    assert data.shape == (4, 20)
    if sio.dataformat in [1, 2, 3, 5, 6, 9, 10, 11, 12]:
        assert data[0, 0] == 8000
        assert data[3, 19] == 11019
    elif sio.dataformat == 8:
        assert data[0, 0] == np.int64(8000).astype(np.int8)
        assert data[3, 19] == np.int64(11019).astype(np.int8)
    elif sio.dataformat == 16:
        assert data[0, 0] == np.int64(8000).astype(np.uint8)
        assert data[3, 19] == np.int64(11019).astype(np.uint8)
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], history=hist)
    assert len(hist) == 1

def test_read_ensemble_noncontiguous(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    sio.create_index(group_by="cdpt", sort_by="cdp")
    ens = sio.read_ensemble(1)
    assert len(ens) == 3

def test_read_ensemble_default_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    sio.create_index(group_by="fldr", sort_by="tracf")
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics=["fldr", "tracf"])
    assert ens["fldr"][0] == 3
    assert ens["tracf"][-1] == 4
    with pytest.raises(ValueError):
        ens["trid"]
    data = ens["data"]
    assert data.shape == (4, 20)
    if sio.dataformat in [1, 2, 3, 5, 6, 9, 10, 11, 12]:
        assert data[0, 0] == 8000
        assert data[3, 19] == 11019
    elif sio.dataformat == 8:
        assert data[0, 0] == np.int64(8000).astype(np.int8)
        assert data[3, 19] == np.int64(11019).astype(np.int8)
    elif sio.dataformat == 16:
        assert data[0, 0] == np.int64(8000).astype(np.uint8)
        assert data[3, 19] == np.int64(11019).astype(np.uint8)
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics=["fldr", "tracf"], history=hist)
    assert len(hist) == 1

def test_read_ensemble_order(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    sio.create_index(group_by="fldr", sort_by="tracf", group_order="<", sort_order="<")
    ens = sio.read_ensemble(sio.ensemble_keys[-1])
    assert ens["fldr"][0] == 1
    assert ens["tracf"][-1] == 1
    data = ens["data"]
    assert data.shape == (4, 20)
    assert data[3, 19] == 19
    if sio.dataformat in [1, 2, 3, 5, 6, 9, 10, 11, 12]:
        assert data[0, 0] == 3000
    elif sio.dataformat == 8:
        assert data[0, 0] == np.int64(3000).astype(np.int8)
    elif sio.dataformat == 16:
        assert data[0, 0] == np.int64(3000).astype(np.uint8)
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], history=hist)
    assert len(hist) == 1

def test_read_ensemble_order_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    sio.create_index(group_by="fldr", sort_by="tracf", group_order="<", sort_order="<")
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics=["fldr", "tracf"])
    assert ens["fldr"][0] == 1
    assert ens["tracf"][-1] == 1
    with pytest.raises(ValueError):
        ens["trid"]
    data = ens["data"]
    assert data.shape == (4, 20)
    assert data[3, 19] == 19
    if sio.dataformat in [1, 2, 3, 5, 6, 9, 10, 11, 12]:
        assert data[0, 0] == 3000
    elif sio.dataformat == 8:
        assert data[0, 0] == np.int64(3000).astype(np.int8)
    elif sio.dataformat == 16:
        assert data[0, 0] == np.int64(3000).astype(np.uint8)
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics="fldr", history=hist)
    assert len(hist) == 1

def test_read_ensemble_filt(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    def filt_func(x): return (x["fldr"] < 3)
    sio.create_index(group_by="fldr", sort_by="tracf", filt=filt_func)
    ens = sio.read_ensemble(sio.ensemble_keys[-1])
    assert ens["fldr"][0] == 2
    assert ens["tracf"][-1] == 4
    data = ens["data"]
    assert data.shape == (4, 20)
    if sio.dataformat in [1, 2, 3, 5, 6, 9, 10, 11, 12]:
        assert data[0, 0] == 4000
        assert data[3, 19] == 7019
    elif sio.dataformat == 8:
        assert data[0, 0] == np.int64(4000).astype(np.int8)
        assert data[3, 19] == np.int64(7019).astype(np.int8)
    elif sio.dataformat == 16:
        assert data[0, 0] == np.int64(4000).astype(np.uint8)
        assert data[3, 19] == np.int64(7019).astype(np.uint8)
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], history=hist)
    assert len(hist) == 1

def test_read_ensemble_filt_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    def filt_func(x): return (x["fldr"] < 3)
    sio.create_index(group_by="fldr", sort_by="tracf", filt=filt_func)
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics=["fldr", "tracf"])
    assert ens["fldr"][0] == 2
    assert ens["tracf"][-1] == 4
    with pytest.raises(ValueError):
        ens["trid"]
    data = ens["data"]
    assert data.shape == (4, 20)
    if sio.dataformat in [1, 2, 3, 5, 6, 9, 10, 11, 12]:
        assert data[0, 0] == 4000
        assert data[3, 19] == 7019
    elif sio.dataformat == 8:
        assert data[0, 0] == np.int64(4000).astype(np.int8)
        assert data[3, 19] == np.int64(7019).astype(np.int8)
    elif sio.dataformat == 16:
        assert data[0, 0] == np.int64(4000).astype(np.uint8)
        assert data[3, 19] == np.int64(7019).astype(np.uint8)
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics=["fldr", "tracf"], history=hist)
    assert len(hist) == 1

def test_read_ensemble_multi(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    sio.create_index(group_by=["fldr", "cdp"], sort_by=["tracf"])
    ens = sio.read_ensemble(sio.ensemble_keys[-1])
    assert ens["fldr"][0] == 3
    assert ens["tracf"][-1] == 4
    data = ens["data"]
    assert data.shape == (4, 20)
    if sio.dataformat in [1, 2, 3, 5, 6, 9, 10, 11, 12]:
        assert data[0, 19] == 8019
    elif sio.dataformat == 8:
        assert data[0, 19] == np.int64(8019).astype(np.int8)
    elif sio.dataformat == 16:
        assert data[0, 19] == np.int64(8019).astype(np.uint8)
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], history=hist)
    assert len(hist) == 1

def test_read_ensemble_multi_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    sio.create_index(group_by=["fldr", "cdp"], sort_by=["tracf"])
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics=["fldr", "tracf"])
    assert ens["fldr"][0] == 3
    assert ens["tracf"][-1] == 4
    with pytest.raises(ValueError):
        ens["trid"]
    data = ens["data"]
    assert data.shape == (4, 20)
    if sio.dataformat in [1, 2, 3, 5, 6, 9, 10, 11, 12]:
        assert data[0, 19] == 8019
    elif sio.dataformat == 8:
        assert data[0, 19] == np.int64(8019).astype(np.int8)
    elif sio.dataformat == 16:
        assert data[0, 19] == np.int64(8019).astype(np.uint8)
    hist = []
    ens = sio.read_ensemble(sio.ensemble_keys[-1], mnemonics=["fldr", "tracf"], history=hist)
    assert len(hist) == 1

def test_read_ensembles(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    with pytest.raises(RuntimeError):
        for ens in sio.ensembles():
            pass

    hist = []
    sio.create_index(group_by="fldr", sort_by="tracf")
    with np.errstate(all='ignore'):
        for i, ens in enumerate(sio.ensembles(history=hist)):
            assert ens["fldr"][0] == 1+i
            assert ens["tracl"][0] == 1+i*4
            data = ens["data"]
            assert data.shape == (4, 20)
            if sio.dataformat in [1, 2, 3, 5, 6, 9, 10, 11, 12]:
                assert data[0, 19] == i*4000+19
            elif sio.dataformat == 8:
                assert data[0, 19] == np.int64(i*4000+19).astype(np.int8)
            elif sio.dataformat == 16:
                assert data[0, 19] == np.uint64(i*4000+19).astype(np.uint8)
            assert len(hist) == i+1

def test_read_ensembles_mnemonics(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    sio.create_index(group_by="fldr", sort_by="tracf")
    hist = []
    for i, ens in enumerate(sio.ensembles(mnemonics=["fldr", "tracl"], history=hist)):
        assert ens["fldr"][0] == 1+i
        assert ens["tracl"][0] == 1+i*4
        with pytest.raises(ValueError):
            ens["trid"]
        data = ens["data"]
        assert data.shape == (4, 20)
        if sio.dataformat in [1, 2, 3, 5, 6, 9, 10, 11, 12]:
            assert data[0, 19] == i*4000+19
        elif sio.dataformat == 8:
            assert data[0, 19] == np.int64(i*4000+19).astype(np.int8)
        elif sio.dataformat == 16:
            assert data[0, 19] == np.uint64(i*4000+19).astype(np.uint8)
        assert len(hist) == i+1

def test_read_getitem(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)

    nt = len(sio)
    assert nt == 12

    dataset = sio[0]
    assert dataset["tracl"][0] == 1

    dataset = sio[0:2]
    assert dataset["tracl"][0] == 1
    assert dataset["tracl"][1] == 2

    dataset = sio[[0, 1, 2]]
    assert dataset["tracl"][0] == 1
    assert dataset["tracl"][2] == 3

    dataset = sio[[0, 1, 2, 1, 0]]
    assert len(dataset) == 3

    dataset = sio[1:nt-1:1]
    assert len(dataset) == nt-2

    dataset = sio[::]
    assert len(dataset) == nt
    assert dataset["tracl"][0] == 1
    assert dataset["tracl"][-1] == 12

    dataset = sio[::-1]
    assert len(dataset) == nt
    assert dataset["tracl"][0] == 12
    assert dataset["tracl"][-1] == 1

    dataset = sio[:1000:]
    assert len(dataset) == nt

    dataset = sio[::2]
    assert len(dataset) == nt//2
    assert dataset["tracl"][0] == 1
    assert dataset["tracl"][-1] == 11

    dataset = sio[-1:None:-2]
    assert len(dataset) == nt//2
    assert dataset["tracl"][0] == 11
    assert dataset["tracl"][-1] == 1

    dataset = sio[-1:None:-1]
    assert len(dataset) == nt
    assert dataset["tracl"][0] == 12
    assert dataset["tracl"][-1] == 1

    dataset = sio[None:None:-2]
    assert len(dataset) == nt//2
    assert dataset["tracl"][0] == 11
    assert dataset["tracl"][-1] == 1

def test_thstat_default(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
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

def test_thstat_memory(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    h = sio.read_all_headers()
    thstat1 = sio.thstat(headers=h)
    assert thstat1 is not None
    thstat2 = sio.thstat(headers=h, ntmax=10)
    assert thstat2 is not None
    thstat3 = sio.thstat(headers=h, ntmax=99)
    assert thstat3 is not None

def test_log_thstat(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    dataset = sio.read_dataset()
    thstat1 = seisio.log_thstat(dataset, zero=False)
    assert thstat1 is not None
    thstat2 = seisio.log_thstat(dataset, zero=True)
    assert thstat2 is not None
    thstat3 = sio.log_thstat()
    assert thstat3 is not None
    thstat4 = sio.log_thstat(traces=dataset)
    assert thstat4 is not None
    thstat5 = sio.log_thstat(thstat=thstat1, traces=dataset, zero=True)
    assert thstat5 is not None


def test_log_thstat_patch(dummy_segy_file, monkeypatch):
    monkeypatch.delitem(sys.modules, "tabulate", raising=False)
    monkeypatch.setitem(sys.modules, "tabulate", None)
    sio = seisio.input(dummy_segy_file)
    dataset = sio.read_dataset()
    thstat1 = seisio.log_thstat(dataset, zero=False)
    assert thstat1 is not None
    thstat2= seisio.log_thstat(dataset, zero=True)
    assert thstat2 is not None

def test_open_write(tmp_path):
    file = tmp_path / "dummy_write.segy"
    sol = seisio.output(file, ns=42, vsi=2000, endian="=", unknown_para=42)
    assert sol is not None
    with pytest.raises(ValueError):
        seisio.output(file, ns=42, vsi=2000, endian=">", format=19)
    with pytest.raises(ValueError):
        seisio.output(file, ns=-42, vsi=2000, endian=">")
    with pytest.raises(ValueError):
        seisio.output(file, ns=42, vsi=-2000, endian=">")
    with pytest.raises(ValueError):
        seisio.output(file, ns=42, vsi=2000, endian=">", segymaj=42)
    with pytest.raises(ValueError):
        seisio.output(file, ns=42, vsi=2000, endian=">", segymin=42)
    with pytest.raises(ValueError):
        seisio.output(file, ns=42, vsi=2000, endian=">", ntxtrec=-1)
    with pytest.raises(TypeError):
        seisio.output(file, ns=42, vsi=2000, endian=">", thext1=42)
    with pytest.raises(ValueError):
        seisio.output(file, ns=42, vsi=2000, endian=">", ntxtrail=-1)
    with pytest.raises(ValueError):
        seisio.output(file, ns=42, vsi=2000, endian=">", nthuser=-1)
    sol = seisio.output(file, ns=42, vsi=2000, endian=">", ntxtrec=1, segymaj=0)
    assert sol is not None
    sol = seisio.output(file, ns=42, vsi=2000, endian=">", thext1=True, segymaj=0)
    assert sol is not None
    sol = seisio.output(file, ns=42, vsi=2000, endian=">", ntxtrail=1, segymaj=0)
    assert sol is not None
    sol = seisio.output(file, ns=99999, vsi=2000, endian=">", segymaj=1)
    assert sol is not None

    thdefu = TESTS_DIR / "data" / "my_traceheaders.json"
    # fail fast if file was moved or omitted
    if not thdefu.exists():
        pytest.fail(f"User-defined trace header definition file missing at: {thdefu}")

    sol = seisio.output(file, ns=42, vsi=2000, nthuser=1, thdefu=[thdefu], thext1=False)
    assert sol is not None

    v = sol.nthuser
    assert v == 1

def test_write_little(tmp_path, dummy_segy_file):
    sin = seisio.input(dummy_segy_file)
    assert sin is not None
    data_ref = sin.read_dataset()

    file = tmp_path / f"dummy_out_little_{sin.dataformat}.segy"

    with pytest.raises(ValueError):
        seisio.output(file, ns=sin.ns, endian="XY")

    sol = seisio.output(file, ns=sin.ns, vsi=sin.vsi, endian="<",
                        format=sin.dataformat, segymaj=1, segymin=0,
                        txtenc="ascii", unknown_para=42)
    assert sol is not None

    binh = sol.binhead_template
    assert binh is not None

    txth = sol.txthead_template
    assert txth is not None

    rech = sol.txtrec_template
    assert rech is not None

    sol.init(textual=txth, binary=binh, records=None, unknown="bla")
    sol.init(textual=txth, binary=binh, records=None) # second call, ignored

    sol.log_bhdef()
    sol.log_binhead(binhead=binh)
    sol.log_binhead()

    sol.log_txthead(txthead=txth)
    sol.log_txthead()

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

    sol.finalize()

    sil = seisio.input(file)
    assert sil.nt == 3*sin.nt
    assert sil.ns == 20

    data_rdb = sil.read_dataset()
    assert np.max(data_rdb["ep"]) == 3
    assert np.min(data_rdb["ep"]) == 0
    assert data_rdb["ep"][-1] == 3
    assert np.all(data_rdb["data"][0:sin.nt, :] == data_ref["data"][:, :])
    assert np.all(data_rdb["ep"][2*sin.nt:] == data_ref["fldr"])
    assert np.all(data_rdb["fldr"][sin.nt:2*sin.nt] == data_ref["fldr"])

def test_write_big(tmp_path, dummy_segy_file):
    sin = seisio.input(dummy_segy_file)
    assert sin is not None
    data_ref = sin.read_dataset()

    file = tmp_path / f"dummy_out_big_{sin.dataformat}.segy"

    with pytest.raises(ValueError):
        seisio.output(file, ns=sin.ns, endian="XY")

    sol = seisio.output(file, ns=sin.ns, vsi=sin.vsi, endian=">",
                        format=sin.dataformat, segymaj=1, segymin=0,
                        txtenc="ascii", unknown_para=42)
    assert sol is not None

    binh = sol.binhead_template
    assert binh is not None

    txth = sol.txthead_template
    assert txth is not None

    rech = sol.txtrec_template
    assert rech is not None

    sol.init(textual=None, binary=None, records=None)

    sol.log_bhdef()
    sol.log_binhead(binhead=binh)
    sol.log_binhead()

    sol.log_txthead(txthead=txth)
    sol.log_txthead()

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

    sol.finalize(encode=42)

    sil = seisio.input(file)
    assert sil.nt == 3*sin.nt
    assert sil.ns == 20

    data_rdb = sil.read_dataset()
    assert np.max(data_rdb["ep"]) == 3
    assert np.min(data_rdb["ep"]) == 0
    assert data_rdb["ep"][-1] == 3
    assert np.all(data_rdb["data"][0:sin.nt, :] == data_ref["data"][:, :])
    assert np.all(data_rdb["ep"][2*sin.nt:] == data_ref["fldr"])
    assert np.all(data_rdb["fldr"][sin.nt:2*sin.nt] == data_ref["fldr"])

def test_write_thext1(tmp_path, dummy_segy_file):
    sin = seisio.input(dummy_segy_file)
    assert sin is not None
    data_ref = sin.read_dataset()

    file = tmp_path / f"dummy_out_little_{sin.dataformat}.segy"

    sol = seisio.output(file, ns=sin.ns, vsi=sin.vsi, endian="<",
                        format=sin.dataformat, segymaj=2, segymin=1,
                        txtenc="ascii", thext1=True)
    assert sol is not None

    v = sol.thext1
    assert v == True

    binh = sol.binhead_template
    assert binh is not None

    txth = sol.txthead_template
    assert txth is not None

    sol.init(textual=txth, binary=binh, records=None)

    trac_l = sol.traces_template(nt=sin.nt, headers_only=False)
    assert len(trac_l) == sin.nt
    assert trac_l["data"].shape == (sin.nt, sin.ns)

    nt = sol.write_traces(traces=sin.read_dataset())
    assert nt == sin.nt

    nt += sol.write_traces(data=sin.read_dataset()["data"], headers=sin.read_all_headers())
    assert nt == 2*sin.nt

    nt += sol.write_traces(traces=sin.read_dataset(), remap={"fldr": "ep"})
    assert nt == 3*sin.nt

    sol.finalize()

    sil = seisio.input(file)
    assert sil.nt == 3*sin.nt
    assert sil.ns == 20
    assert sil.thext1 == True

    data_rdb = sil.read_dataset()
    assert np.max(data_rdb["ep"]) == 3
    assert np.min(data_rdb["ep"]) == 0
    assert data_rdb["ep"][-1] == 3
    assert data_rdb["cable"][0] == 0
    assert np.all(data_rdb["data"][0:sin.nt, :] == data_ref["data"][:, :])
    assert np.all(data_rdb["ep"][2*sin.nt:] == data_ref["fldr"])
    assert np.all(data_rdb["fldr"][sin.nt:2*sin.nt] == data_ref["fldr"])

def test_write_txthead(tmp_path, dummy_segy_file):
    sin = seisio.input(dummy_segy_file)
    assert sin is not None
    data_ref = sin.read_dataset()

    file = tmp_path / f"dummy_out_little_{sin.dataformat}.segy"

    sol = seisio.output(file, ns=sin.ns, vsi=sin.vsi, endian="<",
                        format=sin.dataformat, segymaj=2, segymin=1,
                        txtenc="ascii", thext1=True, ntxtrec=2)
    assert sol is not None

    v = sol.thext1
    assert v == True

    v = sol.ntxtrec
    assert v == 2

    binh = sol.binhead_template
    assert binh is not None

    txth = sol.txthead_template
    assert txth is not None

    txrec = sol.txtrec_template
    assert txrec is not None

    rec = [txrec, txrec]

    sol.init(textual=txth, binary=binh, records=rec)

    trac_l = sol.traces_template(nt=sin.nt, headers_only=False)
    assert len(trac_l) == sin.nt
    assert trac_l["data"].shape == (sin.nt, sin.ns)

    nt = sol.write_traces(traces=sin.read_dataset())
    assert nt == sin.nt

    nt += sol.write_traces(data=sin.read_dataset()["data"], headers=sin.read_all_headers())
    assert nt == 2*sin.nt

    nt += sol.write_traces(traces=sin.read_dataset(), remap={"fldr": "ep"})
    assert nt == 3*sin.nt

    sol.finalize()

    sil = seisio.input(file)
    assert sil.nt == 3*sin.nt
    assert sil.ns == 20
    assert sil.thext1 == True
    assert sil.ntxtrec == 2
    assert len(sil.records) == 2
    assert len(sil.get_records()) == 2

    data_rdb = sil.read_dataset()
    assert np.max(data_rdb["ep"]) == 3
    assert np.min(data_rdb["ep"]) == 0
    assert data_rdb["ep"][-1] == 3
    assert data_rdb["cable"][-1] == 0
    assert np.all(data_rdb["data"][0:sin.nt, :] == data_ref["data"][:, :])
    assert np.all(data_rdb["ep"][2*sin.nt:] == data_ref["fldr"])
    assert np.all(data_rdb["fldr"][sin.nt:2*sin.nt] == data_ref["fldr"])

def test_write_txtrail(tmp_path):
    file = tmp_path / "dummy_out_little_txtrail.segy"

    sol = seisio.output(file, ns=99999, vsi=1e-8, endian="<",
                        format=5, segymaj=2, segymin=1,
                        txtenc="ascii", thext1=True, ntxtrec=2, ntxtrail=1)
    assert sol is not None

    v = sol.thext1
    assert v == True

    v = sol.ntxtrec
    assert v == 2

    v = sol.ntxtrail
    assert v == 1

    binh = sol.binhead_template
    assert binh is not None

    txth = sol.txthead_template
    assert txth is not None

    txrec = sol.txtrec_template
    assert txrec is not None

    rec = [txrec, txrec]

    sol.init(textual=txth, binary=binh, records=rec)

    trac_l = sol.traces_template(nt=2, headers_only=False)
    assert len(trac_l) == 2
    assert trac_l["data"].shape == (2, 99999)

    trac_l["data"] = 99
    trac_l["ep"] = [41, 42]

    nt = sol.write_traces(traces=trac_l)
    assert nt == 2

    sol.finalize(txrec, txrec, encode=42) # pass two instead of one, wrong encode value type
    sol.finalize(txrec) # second call, ignored

    sil = seisio.input(file)
    assert sil.nt == 2
    assert sil.ns == 99999
    assert sil.thext1 == True
    assert sil.ntxtrec == 2
    assert sil.ntxtrail == 1
    assert len(sil.records) == 2
    assert len(sil.get_records()) == 2
    assert len(sil.trailers) == 1
    assert len(sil.get_trailers()) == 1

    data_rdb = sil.read_dataset()
    assert np.max(data_rdb["ep"]) == 42
    assert np.min(data_rdb["ep"]) == 41
    assert np.min(data_rdb["data"]) == 99
    assert data_rdb["ep"][-1] == 42
    assert data_rdb["cable"][0] == 0

def test_vslice(dummy_segy_file):
    sio = seisio.input(dummy_segy_file)
    assert sio is not None

    with pytest.raises(ValueError):
        sio.read_vslice(n=-1)
    with pytest.raises(ValueError):
        sio.read_vslice(n=99999)

    sl = sio.read_vslice(n=None)
    assert sl is not None

    hist = []
    sl = sio.read_vslice(n=2, history=hist)
    assert sl is not None
    assert len(hist) == 1

    sl = sio.read_vslice(n=4, reshape=False, silent=True)
    assert sl is not None

def test_write_single_trace(tmp_path, dummy_su_file):
    file_path = tmp_path / "dummy_out_single.segy"

    sin = seisio.input(dummy_su_file)
    assert sin is not None
    dataset = sin.read_traces(0)
    
    sol = seisio.output(file_path, ns=sin.ns, vsi=sin.vsi)
    assert sol is not None
    
    sol.init()
    
    nt_written = sol.write_traces(traces=dataset)
    assert nt_written == 1
    
    headers = tools.remove_mnemonic(dataset, names=["data"])
    assert headers is not None
    
    df = pd.DataFrame(headers)
    nt_written += sol.write_traces(data=dataset["data"], headers=df)
    assert nt_written == 2
    
    sol.finalize()
    
    sin_check = seisio.input(file_path)
    assert sin_check.ns == sin.ns
    assert sin_check.nt == 2
