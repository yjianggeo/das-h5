import numpy as np
from obspy import UTCDateTime

from das_h5.psd.compute import das_psd


def test_das_psd_peaks_at_input_frequency():
    delta = 0.01           # 100 Hz sampling
    n = 2000               # 20 s
    f0 = 10.0
    t = np.arange(n) * delta
    data = np.sin(2 * np.pi * f0 * t)[:, None].astype(np.float64)

    fss, psds, times, maxv = das_psd(
        das_data=data, delta=delta, wind_len=2.0, wind_step=2.0,
        minfreq=[1.0], maxfreq=[40.0], ch1=0,
        starttime=UTCDateTime(0), scaling="density")

    freqs = fss[0][0]            # band 0 frequency axis of first segment
    power = psds[0][0][:, 0]     # band 0 PSD of channel 0
    peak_freq = freqs[int(np.argmax(power))]
    assert abs(peak_freq - f0) < 1.0
    assert len(times) >= 1
