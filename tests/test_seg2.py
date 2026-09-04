
import numpy as np
import pytest
import seisio

from pathlib import Path

def test_open(sample_seg2_file):
    sio = seisio.input(sample_seg2_file)
    assert sio is not None

    v = sio.dataformat
    assert v >0 and v <= 5

    end = sio.endianess
    assert end in [">", "<"]

    v = sio.file
    assert v == Path(sample_seg2_file)

    v = sio.fsize
    assert v > 0

    v = sio.mnemonics
    assert "CDP_NUMBER" in v
    assert "SAMPLE_INTERVAL" in v
    assert "TRACE_TYPE" in v

    v = sio.ns
    assert v == 1000

    v = sio.nsamples
    assert v == 1000

    v = sio.nt
    assert v == 48

    v = sio.ntraces
    assert v == 48

    v = len(sio)
    assert v == 48

    thsize = sio.thsize
    assert thsize >= 240

    v = sio.trsize
    assert v > thsize

    v = sio.log_thdef()
    assert v is None

    v = sio.vsi
    assert v == 0.00025

    v = sio.fheader
    assert v is not None
    assert isinstance(v, dict)
    assert v["JOB_ID"]
    assert v["TRACE_SORT"]

def test_read_all_headers(sample_seg2_file):
    sio = seisio.input(sample_seg2_file)
    h = sio.read_all_headers()
    assert len(h) == 48
    assert h["ALIAS_FILTER"][0] == "1666.66 0"
    assert "47.0" in h["RECEIVER_LOCATION"][47]

def test_read_all_traces(sample_seg2_file):
    sio = seisio.input(sample_seg2_file)
    data, headers = sio.read_all_traces()
    assert len(headers) == 48
    assert data.shape == (48, 1000)
    assert "47.0" in headers["RECEIVER_LOCATION"][47]
    assert np.max(np.abs(data)) > 0
    hist = []
    data, headers = sio.read_all_traces(history=hist)
    assert len(hist) == 1

def test_read_dataset(sample_seg2_file):
    sio = seisio.input(sample_seg2_file)
    data, headers = sio.read_all_traces()
    assert len(headers) == 48
    assert data.shape == (48, 1000)
    assert "47.0" in headers["RECEIVER_LOCATION"][47]
    assert np.max(np.abs(data)) > 0
    hist = []
    data, headers = sio.read_all_traces(history=hist)
    assert len(hist) == 1

def test_open_wrongtype(dummy_su_file):
    with pytest.raises(ValueError):
        seisio.input(dummy_su_file, filetype="SEG2")

def test_read_all_traces_notfixed(sample_seg2_file):
    sio = seisio.input(sample_seg2_file)
    sio._fp.fixed = False
    sio._dp.ns = 999
    data, headers = sio.read_all_traces()
    assert len(headers) == 48
    assert data.shape == (48, 1000)
    assert "47.0" in headers["RECEIVER_LOCATION"][47]
    assert np.max(np.abs(data)) > 0
