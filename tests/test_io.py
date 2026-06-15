import numpy as np
import h5py
from obspy import UTCDateTime

from das_h5.io.h5_reader import make_dbfile, read_h5
from das_h5.io.writer import save_npz


def _make_fake_h5(path, start_utc, npts=1000, nch=5, rate=1000.0, gauge=10.0):
    t0_us = int(start_utc.timestamp * 1e6)
    with h5py.File(path, "w") as h5:
        raw = h5.create_group("Acquisition/Raw[0]")
        data = (np.arange(npts)[:, None] * np.ones((1, nch))).astype(np.float32)
        raw.create_dataset("RawData", data=data)
        raw.create_dataset("RawDataTime", data=np.array([t0_us], dtype=np.int64))
        raw.attrs["OutputDataRate"] = rate
        h5["Acquisition"].attrs["GaugeLength"] = gauge


def test_make_dbfile_and_read_h5_roundtrip(tmp_path):
    start = UTCDateTime("2024-01-01T00:00:00")
    _make_fake_h5(str(tmp_path / "das_000.h5"), start, npts=1000, nch=5, rate=1000.0)

    db = make_dbfile(str(tmp_path))
    assert len(db) == 1

    data, dt = read_h5(db_list=db, time_flag=start, begin_time=0, end_time=0.5,
                       channel_begin=0, channel_end=4)
    assert data.shape[1] == 5
    assert abs(dt - 0.001) < 1e-9
    assert 490 <= data.shape[0] <= 510

    # strain-rate conversion: raw * 107.9 * 1e-9 * frequency / gauge_length,
    # with frequency = OutputDataRate (1000 Hz) and gauge_length = 10 m.
    # The synthetic RawData has value k in row k, so row 1 == 1 * 107.9e-9 * 1000 / 10.
    assert np.isclose(data[1, 0], 1.0 * 107.9 * 1e-9 * 1000.0 / 10.0)


def test_save_npz_roundtrip(tmp_path):
    data = np.random.randn(200, 4).astype(np.float32)
    out = str(tmp_path / "trace.npy")
    save_npz(data, outnpy=out)
    reloaded = np.load(out)
    assert reloaded.shape == data.shape
    assert np.allclose(reloaded, data)
