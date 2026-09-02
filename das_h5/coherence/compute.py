"""Poincare spectral-coherence coupling analysis for DAS data.

Implements the four-stage framework of Shiri & Belal (2026), "A real-time
framework for mapping subsea cable burial state using Poincare spectral
coherence of DAS measurements" (Scientific Reports 16:23647):

    Stage 1  two-band spectral fingerprints from band-limited energy (FBE)
    Stage 2  block-wise Poincare-style spatial coherence C_eff
    Stage 3  robust log-MAD normalisation
    Stage 4  spatial Gaussian smoothing and temporal median -> BI profile

plus a candidate band-pair scan (`optimize_band_pair`) that scores band pairs
by spatial contrast x temporal (split-half) stability.

The metric is unit-independent: the spectral fingerprint is a normalised
ratio of two band powers, so any constant calibration factor cancels out.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

EPS1, EPS2, EPS3 = 1e-30, 1e-12, 1e-12


def das_fbe(das_data, delta, wind_len, wind_step, band):
    '''
    Frequency Band Energy (FBE): band-integrated spectral power of every
    channel in every sliding window (Eq. 6 of Shiri & Belal 2026).
    By yjianggeo 2026.09.02

    :param das_data:    das data array [times, channels]
    :param delta:       time delta [s]
    :param wind_len:    window length [s]
    :param wind_step:   window hop [s]
    :param band:        (f0, f1) frequency band [Hz]
    :return:            times [s, window centres], power [nwin, nch]
    '''
    nsamp = int(round(wind_len / delta))
    nstep = int(round(wind_step / delta))
    hann = np.hanning(nsamp)
    fs = 1.0 / delta
    k0 = int(np.ceil(band[0] * nsamp / fs))
    k1 = int(np.floor(band[1] * nsamp / fs)) + 1
    k0 = max(1, k0)                          # avoid using the offset
    k1 = min(nsamp // 2 + 1, k1)             # avoid exceeding nyquist
    starts = np.arange(0, das_data.shape[0] - nsamp + 1, nstep)
    power = np.empty((len(starts), das_data.shape[1]), dtype=np.float64)
    for j, s in enumerate(starts):
        seg = das_data[s:s + nsamp].astype(np.float64)
        seg -= seg.mean(axis=0)              # remove DC before taper
        spec = np.fft.rfft(seg * hann[:, None], axis=0)
        power[j] = (np.abs(spec[k0:k1]) ** 2).sum(axis=0)
    times = (starts + nsamp / 2.0) * delta
    return times, power


def poincare_coherence_map(p_low, p_high, block=10, block_step=None, alpha=0.05):
    '''
    Block-wise Poincare-style spatial coherence C_raw[w, b] from the two-band
    spectral fingerprints (Eqs. 8-19 of Shiri & Belal 2026).
    By yjianggeo 2026.09.02

    :param p_low:       low-band FBE [nwin, nch]
    :param p_high:      high-band FBE [nwin, nch]
    :param block:       channels per block M
    :param block_step:  block step in channels (default = block, non-overlap)
    :param alpha:       gradient-floor factor (Eq. 16)
    :return:            c_raw [nwin, nblk], block_begin [nblk] first channels
    '''
    if block_step is None:
        block_step = block
    s = p_low + p_high + EPS1                                    # Eq. 7
    f = np.stack([p_low / s, p_high / s], axis=-1)               # Eq. 8
    nwin, nch = f.shape[:2]
    nblk = (nch - block) // block_step + 1
    c_raw = np.empty((nwin, nblk), dtype=np.float64)
    for b in range(nblk):
        sl = slice(b * block_step, b * block_step + block)
        fb = f[:, sl]                                            # (nwin, M, 2)
        fbar = fb.mean(axis=1, keepdims=True)                    # Eq. 11
        spread = np.sqrt(((fb - fbar) ** 2).sum(axis=(1, 2)) / block)        # Eq. 12
        grad = np.sqrt((np.diff(fb, axis=1) ** 2).sum(axis=(1, 2))
                       / (block - 1))                            # Eqs. 13-14
        scale = np.sqrt((fb ** 2).sum(axis=(1, 2)) / block)      # Eq. 15
        grad_eff = np.maximum(grad, alpha * scale)               # Eqs. 16-17
        c_raw[:, b] = spread / (grad_eff + EPS2)                 # Eq. 18
    block_begin = np.arange(nblk) * block_step
    return c_raw, block_begin                                    # Eq. 19


def robust_normalise(c_raw):
    '''
    Robust log-MAD normalisation of raw coherence map(s) (Eqs. 20-23).
    A list of maps is normalised jointly (shared median/MAD), which keeps
    absolute levels comparable e.g. between time periods.
    By yjianggeo 2026.09.02

    :param c_raw:   one [nwin, nblk] map or a list of maps
    :return:        normalised map(s) in the same form
    '''
    single = not isinstance(c_raw, (list, tuple))
    maps = [c_raw] if single else list(c_raw)
    log_all = np.log(np.concatenate([m.ravel() for m in maps]) + EPS3)      # Eq. 20
    m_l = np.median(log_all)
    mad_l = np.median(np.abs(log_all - m_l))                                # Eq. 21
    out = [np.exp((np.log(m + EPS3) - m_l) / (1.4826 * mad_l + EPS3))
           for m in maps]                                       # Eqs. 22-23
    return out[0] if single else out


def smooth_bi(c_norm, smooth_scale=0.75):
    '''
    Spatial Gaussian smoothing of the normalised coherence field and temporal
    aggregation into the Burial Index (BI) profile (Eqs. 24-26).
    By yjianggeo 2026.09.02

    :param c_norm:      normalised coherence map [nwin, nblk]
    :param smooth_scale: Gaussian sigma in channel samples
                        (sigma = L_smooth / channel_dx; 0.75 = 3 m at 4 m
                        spacing, the value used in the paper)
    :return:            c_smooth [nwin, nblk], bi [nblk]
    '''
    c_smooth = gaussian_filter1d(c_norm, smooth_scale, axis=1, mode='reflect')  # Eq. 24
    bi = np.median(c_smooth, axis=0)                                        # Eq. 26
    return c_smooth, bi


def das_bi(das_data, delta, bands, wind_len=2.0, wind_step=1.0, block=10,
           block_step=None, alpha=0.05, smooth_scale=0.75):
    '''
    Full Poincare spectral-coherence pipeline: DAS data -> BI profile.
    Default parameters follow Algorithm 1 of Shiri & Belal (2026).
    By yjianggeo 2026.09.02

    :param das_data:    das data array [times, channels]
    :param delta:       time delta [s]
    :param bands:       ((low_f0, low_f1), (high_f0, high_f1)) [Hz]
    :param wind_len:    window length [s]
    :param wind_step:   window hop [s]
    :param block:       channels per block M
    :param block_step:  block step in channels (default = block)
    :param alpha:       gradient-floor factor
    :param smooth_scale: Gaussian smoothing sigma in channel samples
    :return:            dict with times, block_channels, c_raw, c_norm,
                        c_smooth, bi, bands, params
    '''
    times, p_low = das_fbe(das_data, delta, wind_len, wind_step, bands[0])
    _, p_high = das_fbe(das_data, delta, wind_len, wind_step, bands[1])
    c_raw, block_begin = poincare_coherence_map(p_low, p_high, block=block,
                                                block_step=block_step,
                                                alpha=alpha)
    c_norm = robust_normalise(c_raw)
    c_smooth, bi = smooth_bi(c_norm, smooth_scale)
    if block_step is None:
        block_step = block
    return {'times': times, 'block_channels': block_begin,
            'c_raw': c_raw, 'c_norm': c_norm, 'c_smooth': c_smooth,
            'bi': bi, 'bands': bands,
            'params': {'wind_len': wind_len, 'wind_step': wind_step,
                       'block': block, 'block_step': block_step,
                       'alpha': alpha, 'smooth_scale': smooth_scale}}


def optimize_band_pair(das_data, delta, wind_len=2.0, wind_step=1.0, block=10,
                       alpha=0.05, smooth_scale=0.75,
                       low_cands=None, high_cands=None,
                       low_range=(0.5, 12.0), low_bin=1.0,
                       high_range=(15.0, 250.0), high_bin=5.0):
    '''
    Scan candidate (low, high) band pairs and score each by

        score = spatial contrast x split-half stability

    where contrast is the spatial std of log(BI) along the cable and
    stability is the Spearman correlation between the BI profiles computed
    from the first and second halves of the time series. Band powers are
    pre-binned (narrow bins), so every candidate pair only sums bins and
    re-runs the cheap block statistics.
    By yjianggeo 2026.09.02

    :param das_data:    das data array [times, channels]
    :param delta:       time delta [s]
    :param wind_len:    window length [s]
    :param wind_step:   window hop [s]
    :param block:       channels per block M
    :param alpha:       gradient-floor factor
    :param smooth_scale: Gaussian smoothing sigma in channel samples
    :param low_cands:   low-band candidates [(f0, f1), ...], default:
                        starts 0.5/1/2/3 Hz x ends 5/8/10/12 Hz (width >= 2 Hz)
    :param high_cands:  high-band candidates [(f0, f1), ...], default:
                        starts 15..230 Hz (5 Hz grid) x widths 20/30/40 Hz
    :param low_range:   (f0, f1) of the 1 Hz bin grid below the high range
    :param low_bin:     low-range bin width [Hz]
    :param high_range:  (f0, f1) of the 5 Hz bin grid
    :param high_bin:    high-range bin width [Hz]
    :return:            best_pair ((l0, l1), (h0, h1)), scores sorted desc;
                        each row: (l0, l1, h0, h1, contrast, stability, score)
    '''
    if low_cands is None:
        low_cands = [(s, e) for s in (0.5, 1.0, 2.0, 3.0)
                     for e in (5.0, 8.0, 10.0, 12.0) if e - s >= 2.0]
    if high_cands is None:
        high_cands = [(float(s), float(s + w))
                      for s in range(int(high_range[0]), int(high_range[1]) - 19, 5)
                      for w in (20, 30, 40) if s + w <= high_range[1]]

    # binned band powers for every window/channel
    edges = np.concatenate([np.arange(low_range[0], low_range[1] + 0.5 * low_bin, low_bin),
                            np.arange(high_range[0], high_range[1] + 0.5 * high_bin, high_bin)])
    nsamp = int(round(wind_len / delta))
    nstep = int(round(wind_step / delta))
    fs = 1.0 / delta
    hann = np.hanning(nsamp)
    bin_idx = [(max(1, int(np.ceil(e0 * nsamp / fs))),
                min(nsamp // 2 + 1, int(np.floor(e1 * nsamp / fs)) + 1))
               for e0, e1 in zip(edges[:-1], edges[1:])]
    starts = np.arange(0, das_data.shape[0] - nsamp + 1, nstep)
    binned = np.empty((len(starts), das_data.shape[1], len(bin_idx)),
                      dtype=np.float32)
    for j, s in enumerate(starts):
        seg = das_data[s:s + nsamp].astype(np.float64)
        seg -= seg.mean(axis=0)
        psd = np.abs(np.fft.rfft(seg * hann[:, None], axis=0)) ** 2
        for b, (k0, k1) in enumerate(bin_idx):
            binned[j, :, b] = psd[k0:k1].sum(axis=0)

    def band_power(e0, e1):
        i0 = int(np.searchsorted(edges, e0 - 1e-9))
        i1 = int(np.searchsorted(edges, e1 - 1e-9))
        return binned[:, :, i0:i1].sum(axis=2).astype(np.float64)

    def bi_of(c_raw):
        c_smooth, bi = smooth_bi(robust_normalise(c_raw), smooth_scale)
        return bi

    scores = []
    for lb in low_cands:
        p_low = band_power(*lb)
        for hb in high_cands:
            c_raw, _ = poincare_coherence_map(p_low, band_power(*hb),
                                              block=block, alpha=alpha)
            half = c_raw.shape[0] // 2
            bi1, bi2 = bi_of(c_raw[:half]), bi_of(c_raw[half:])
            bi_all = bi_of(c_raw)
            contrast = np.log(bi_all + 1e-6).std()
            stab = spearmanr(np.log(bi1 + 1e-6), np.log(bi2 + 1e-6)).statistic
            scores.append((lb[0], lb[1], hb[0], hb[1],
                           float(contrast), float(stab),
                           float(contrast * max(stab, 0.0))))
    scores.sort(key=lambda r: -r[-1])
    best = scores[0]
    best_pair = ((best[0], best[1]), (best[2], best[3]))
    return best_pair, scores
