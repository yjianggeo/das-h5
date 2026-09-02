import numpy as np

from das_h5.coherence.compute import (
    das_fbe, poincare_coherence_map, robust_normalise, smooth_bi, das_bi,
    optimize_band_pair,
)


def test_das_fbe_band_power_peak():
    delta = 0.01           # 100 Hz sampling
    n = 4000               # 40 s
    f0 = 10.0
    t = np.arange(n) * delta
    data = np.sin(2 * np.pi * f0 * t)[:, None].astype(np.float64)

    times, p_in = das_fbe(data, delta, wind_len=2.0, wind_step=1.0,
                          band=(8.0, 12.0))
    _, p_out = das_fbe(data, delta, wind_len=2.0, wind_step=1.0,
                       band=(40.0, 44.0))

    assert p_in.shape == p_out.shape
    assert len(times) == p_in.shape[0]
    assert p_in.mean() > 100 * p_out.mean()


def _synthetic_irregular_block():
    """40 channels; channels 10-19 get a smooth across-channel ramp of
    high-frequency content (large spread vs channel-to-channel step)."""
    rng = np.random.default_rng(0)
    delta = 0.005          # 200 Hz sampling
    n = 12000              # 60 s
    nch = 40
    t = np.arange(n) * delta
    low = np.sin(2 * np.pi * 2.0 * t)[:, None] * np.ones((1, nch))
    hf_amp = np.full(nch, 0.02)
    hf_amp[10:20] = np.linspace(0.05, 1.0, 10)   # smooth ramp inside block 1
    hf = np.sin(2 * np.pi * 60.0 * t)[:, None] * hf_amp[None, :]
    data = low + hf + 0.01 * rng.standard_normal((n, nch))
    return data, delta


def test_bi_marks_irregular_block():
    data, delta = _synthetic_irregular_block()
    res = das_bi(data, delta, bands=((0.5, 5.0), (50.0, 70.0)),
                 wind_len=2.0, wind_step=1.0, block=10, alpha=0.05,
                 channel_dx=1.0, smooth_len=1.0)
    bi = res['bi']
    assert len(bi) == 4                       # 40 channels / block 10
    assert np.all(bi > 0)
    assert int(np.argmax(bi)) == 1            # channels 10-19
    assert res['c_smooth'].shape[1] == 4
    assert res['c_raw'].shape == res['c_norm'].shape


def test_robust_normalise_positive():
    rng = np.random.default_rng(1)
    c = np.exp(rng.standard_normal((20, 5)))
    cn = robust_normalise(c)
    assert cn.shape == c.shape
    assert np.all(cn > 0)
    # joint normalisation of a list returns a list of the same length
    out = robust_normalise([c, 2 * c])
    assert len(out) == 2 and out[1].shape == c.shape


def test_optimize_band_pair_finds_hf_contrast():
    data, delta = _synthetic_irregular_block()
    low_cands = [(0.5, 5.0), (30.0, 40.0)]        # second misses the 2 Hz part
    high_cands = [(50.0, 70.0), (80.0, 95.0)]     # second misses the 60 Hz part
    best, scores = optimize_band_pair(
        data, delta, wind_len=2.0, wind_step=2.0, block=10, alpha=0.05,
        channel_dx=1.0, smooth_len=1.0,
        low_cands=low_cands, high_cands=high_cands,
        low_range=(0.5, 40.0), low_bin=1.0,
        high_range=(45.0, 95.0), high_bin=5.0)
    assert len(scores) == len(low_cands) * len(high_cands)
    assert all(np.isfinite(r[6]) for r in scores)
    assert scores[0][6] >= scores[-1][6]          # sorted desc by score
    # the pair containing both driven bands should win
    assert best == ((0.5, 5.0), (50.0, 70.0))
