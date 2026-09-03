
import numpy as np
import pytest
import seisio

@pytest.mark.slow
def test_input_remote_http_segy():
    file = "http://s3.amazonaws.com/teapot/filt_mig.sgy"
    sio = seisio.input(file)
    assert sio is not None

    v = sio.nt
    assert v == 64860

    v = sio.ns
    assert v == 1501

    h = sio.read_headers(0, sio.nt//2, sio.nt-1)
    assert len(h) == 3

    h = sio.read_headers(0, sio.nt//2, sio.nt-1, mnemonics=["tracl", "tracf", "ns"])
    assert len(h) == 3
    assert len(h.dtype.names) == 3

    h = sio.read_batch_of_headers(start=4, nheaders=5)
    assert len(h) == 5

    h = sio.read_batch_of_headers(start=4, nheaders=5, mnemonics=["tracl", "tracf", "ns"])
    assert len(h) == 5
    assert len(h.dtype.names) == 3

    h = sio.read_multibatch_of_headers(start=0, block=2, count=3, stride=4)
    assert len(h) == 6

    h = sio.read_multibatch_of_headers(start=0, block=2, count=3, stride=4,
                                       mnemonics=["fldr", "tracf", "ns"])
    assert len(h) == 6
    assert len(h.dtype.names) == 3

    h = sio.read_traces(0, 5, 11)
    assert len(h) == 3

    h = sio.read_traces(0, 5, 11, mnemonics=["tracl", "tracf", "ns"])
    assert len(h) == 3
    assert len(h.dtype.names) == 4

    h = sio.read_batch_of_traces(start=4, ntraces=4)
    assert len(h) == 4

    h = sio.read_batch_of_traces(start=60, ntraces=10, mnemonics=["tracl", "tracf", "ns", "dt"])
    assert len(h) == 10
    assert len(h.dtype.names) == 5

    h = sio.read_multibatch_of_traces(start=0, block=2, count=3, stride=4)
    assert len(h) == 6

    h = sio.read_multibatch_of_traces(start=0, block=2, count=3, stride=4,
                                      mnemonics=["fldr", "tracf", "ns"])
    assert len(h) == 6
    assert len(h.dtype.names) == 4

@pytest.mark.slow
def test_input_remote_s3_segy():
    file = "s3://open.source.geoscience/open_data/newzealand/Taranaiki_Basin/PARIHAKA-3D/Parihaka_PSTM_full_angle.sgy"
    sio = seisio.input(file, storage_options={"anon": True})
    assert sio is not None

    v = sio.nt
    assert v == 1038162

    v = sio.ns
    assert v == 1168

    h = sio.read_headers(0, sio.nt//2, sio.nt-1)
    assert len(h) == 3

    h = sio.read_headers(0, sio.nt//2, sio.nt-1, mnemonics=["tracl", "tracf", "ns"])
    assert len(h) == 3
    assert len(h.dtype.names) == 3

    h = sio.read_batch_of_headers(start=4, nheaders=5)
    assert len(h) == 5

    h = sio.read_batch_of_headers(start=4, nheaders=5, mnemonics=["tracl", "tracf", "ns"])
    assert len(h) == 5
    assert len(h.dtype.names) == 3

    h = sio.read_multibatch_of_headers(start=0, block=2, count=3, stride=4)
    assert len(h) == 6

    h = sio.read_multibatch_of_headers(start=0, block=2, count=3, stride=4,
                                       mnemonics=["fldr", "tracf", "ns"])
    assert len(h) == 6
    assert len(h.dtype.names) == 3

    h = sio.read_traces(0, 5, 11)
    assert len(h) == 3

    h = sio.read_traces(0, 5, 11, mnemonics=["tracl", "tracf", "ns"])
    assert len(h) == 3
    assert len(h.dtype.names) == 4

    h = sio.read_batch_of_traces(start=4, ntraces=4)
    assert len(h) == 4

    h = sio.read_batch_of_traces(start=60, ntraces=10, mnemonics=["tracl", "tracf", "ns", "dt"])
    assert len(h) == 10
    assert len(h.dtype.names) == 5

    h = sio.read_multibatch_of_traces(start=0, block=2, count=3, stride=4)
    assert len(h) == 6

    h = sio.read_multibatch_of_traces(start=0, block=2, count=3, stride=4,
                                      mnemonics=["fldr", "tracf", "ns"])
    assert len(h) == 6
    assert len(h.dtype.names) == 4

# @pytest.mark.slow
# def test_input_remote_http_seg2():
#     file = "..."
#     sio = seisio.input(file)
#     assert sio is not None

#     v = sio.dataformat
#     assert v >0 and v <= 5

#     end = sio.endianess
#     assert end in [">", "<"]

#     v = sio.fsize
#     assert v > 0

#     v = sio.mnemonics
#     assert "CDP_NUMBER" in v
#     assert "SAMPLE_INTERVAL" in v
#     assert "TRACE_TYPE" in v

#     v = sio.ns
#     assert v == 1000

#     v = sio.nsamples
#     assert v == 1000

#     v = sio.nt
#     assert v == 48

#     v = sio.ntraces
#     assert v == 48

#     thsize = sio.thsize
#     assert thsize >= 240

#     v = sio.trsize
#     assert v > thsize

#     v = sio.log_thdef()
#     assert v is None

#     v = sio.vsi
#     assert v == 0.00025
    
#     h = sio.read_all_headers()
#     assert len(h) == 48
#     assert h["ALIAS_FILTER"][0] == "1666.66 0"
#     assert "47.0" in h["RECEIVER_LOCATION"][47]
    
#     data, headers = sio.read_all_traces()
#     assert len(headers) == 48
#     assert data.shape == (48, 1000)
#     assert "47.0" in headers["RECEIVER_LOCATION"][47]
#     assert np.max(np.abs(data)) > 0
