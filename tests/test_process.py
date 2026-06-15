import numpy as np

from das_h5.process.filters import nextpow2, norm_data, detrend, filter
from das_h5.process.resample import decimate


def test_nextpow2():
    assert nextpow2(100) == 128
    assert nextpow2(128) == 128


def test_norm_data_peak_is_one():
    data = np.array([[2.0, -4.0], [1.0, 2.0]])
    out = norm_data(data.copy())
    assert np.allclose(np.max(np.abs(out), axis=0), 1.0)


def test_detrend_removes_linear_trend():
    n = 1000
    trend = (3.0 * np.arange(n) + 5.0)[:, None]
    out = detrend(trend.copy().astype(float), type="linear")
    assert np.allclose(out, 0.0, atol=1e-6)


def test_bandpass_suppresses_out_of_band():
    delta = 0.01  # 100 Hz sampling
    n = 4096
    t = np.arange(n) * delta
    inband = np.sin(2 * np.pi * 2.0 * t)    # 2 Hz, kept
    outband = np.sin(2 * np.pi * 40.0 * t)  # 40 Hz, rejected
    data = (inband + outband)[:, None]
    out = filter(data.copy(), delta=delta, type="bandpass",
                 freq=(1.0, 10.0), order=4, zerophase=True)
    # residual after removing the in-band part is far smaller than the out-of-band input
    assert np.std(out[:, 0] - inband) < 0.2 * np.std(outband)


def test_decimate_changes_rate_and_length():
    delta = 0.01
    n = 1000
    data = np.random.randn(n, 3)
    out = decimate(data, delta=delta, factor=5)
    assert out.shape == (n // 5, 3)
