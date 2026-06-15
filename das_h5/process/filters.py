"""DAS array processing: normalization, detrend, and IIR filtering.

Repackaged from DASh5 das_processing.py (By Hardy Hong). Algorithms unchanged.
"""

import numpy as np
import warnings

def nextpow2(N):
    """ Function for finding the next power of 2 """
    n = 1
    while n < N:
        n *= 2
    return n

def norm_data(data, type='self'):
    '''
    Data normalization (self or all).
    By Hardy Hong 2023.07.01
    :param data: h5 data np.array
    :param type: self or all
    :return:     normalized data
    '''

    if type == 'self':
        selfmax_values = np.max(abs(data), axis=0)
        data = data / selfmax_values
    elif type == 'all':
        selfmax_values = np.max(np.max(abs(data), axis=0))
        data = data / selfmax_values
    return data

def rmean(data, axis=0):
    '''
    Remove data.mean.
    By Hardy Hong 2023.07.12

    :param data:   data array
    :param axis:   which axis to rmean
    :return:       data array
    '''

    data -= data.mean(axis=axis)

def detrend(data, type='constant', axis=0):
    '''
    Detrend the h5 data. (simple model)
    By Hardy Hong 2023.07.11

    :param data: h5_data
    :param type: which type to detrend
    :return:     h5_data of detrend
    '''
    from scipy.signal import detrend

    if type == 'simple':
        ndat = len(data[:, 0])
        x1, x2 = data[0, :], data[-1:, ]
        trend = x1 + np.arange(ndat) * (x2 - x1) / float(ndat - 1)
        data -= trend
    if type == 'constant' or type == 'linear':
        data = detrend(data, axis=axis, type=type)
    return data

def filter(data, delta=None, type=None, axis=0, **kwargs):
    '''
    To filter the h5 data.
    By Hardy Hong 2023.07.11

    :param data:  data array
    :param delta: delta time
    :param type:  bandpass, bandstop, highpass, lowpass
    :param kwargs: freq, order, zerophase
    :return: data array with filtering
    '''

    from scipy.signal import iirfilter, cheb2ord, cheby2
    try:
        from scipy.signal import sosfilt
        from scipy.signal import zpk2sos
    except ImportError:
        print('Import errors!')


    df = 1 / delta
    fe = df / 2
    corners = kwargs.get('order', 4)
    zerophase = kwargs.get('zerophase', None)



    if type == 'bandpass':
        freqmin, freqmax = kwargs.get('freq', (None, None))
        if freqmax and freqmin:
            low = freqmin / fe
            high = freqmax / fe
            if high - 1.0 > -1e-6:
                msg = ("Selected high corner frequency ({}) of bandpass is at or "
                       "above Nyquist ({}). Applying a high-pass instead.").format(
                    freqmax, fe)
                warnings.warn(msg)

            if low > 1:
                msg = "Selected low corner frequency is above Nyquist."
                raise ValueError(msg)
            z, p, k = iirfilter(corners, [low, high], btype='band',
                                ftype='butter', output='zpk')
            sos = zpk2sos(z, p, k)
            if zerophase:
                firstpass = sosfilt(sos, data, axis=axis)
                return sosfilt(sos, firstpass[::-1, :], axis=axis)[::-1, :]
            else:
                return sosfilt(sos, data, axis=axis)

    if type == 'bandstop':
        freqmin, freqmax = kwargs.get('freq', (None, None))
        if freqmax and freqmin:
            low = freqmin / fe
            high = freqmax / fe
            if high > 1:
                high = 1.0
                msg = "Selected high corner frequency is above Nyquist. " + \
                      "Setting Nyquist as high corner."
                warnings.warn(msg)
            if low > 1:
                msg = "Selected low corner frequency is above Nyquist."
                raise ValueError(msg)
            z, p, k = iirfilter(corners, [low, high],
                                btype='bandstop', ftype='butter', output='zpk')
            sos = zpk2sos(z, p, k)
            if zerophase:
                firstpass = sosfilt(sos, data, axis=axis)
                return sosfilt(sos, firstpass[::-1, :], axis=axis)[::-1, :]
            else:
                return sosfilt(sos, data, axis=axis)

    if type == 'lowpass':
        freq = kwargs.get('freq', None)
        df = 1 / delta
        if freq:
            f = freq / fe
            # raise for some bad scenarios
            if f > 1:
                f = 1.0
                msg = "Selected corner frequency is above Nyquist. " + \
                      "Setting Nyquist as high corner."
                warnings.warn(msg)
            z, p, k = iirfilter(corners, f, btype='lowpass', ftype='butter',
                                output='zpk')
            sos = zpk2sos(z, p, k)
            if zerophase:
                firstpass = sosfilt(sos, data, axis=axis)
                return sosfilt(sos, firstpass[::-1, :], axis=axis)[::-1, :]
            else:
                return sosfilt(sos, data, axis=axis)

    if type == 'highpass':
        freq = kwargs.get('freq', None)
        fe = 0.5 * df
        f = freq / fe
        # raise for some bad scenarios
        if f > 1:
            msg = "Selected corner frequency is above Nyquist."
            raise ValueError(msg)
        z, p, k = iirfilter(corners, f, btype='highpass', ftype='butter',
                            output='zpk')
        sos = zpk2sos(z, p, k)
        if zerophase:
            firstpass = sosfilt(sos, data, axis=axis)
            return sosfilt(sos, firstpass[::-1, :], axis=axis)[::-1, :]
        else:
            return sosfilt(sos, data, axis=axis)

    if type == 'lowpass_cheby_2':
        maxorder = kwargs.get('maxorder', 16)
        freq = kwargs.get('freq', 1)
        ba = kwargs.get('ba', False)
        freq_passband = kwargs.get('freq_passband', False)
        nyquist = df * 0.5
        # rp - maximum ripple of passband, rs - attenuation of stopband
        rp, rs, order = 1, 96, 1e99
        ws = freq / nyquist  # stop band frequency
        wp = ws  # pass band frequency
        # raise for some bad scenarios
        if ws > 1:
            ws = 1.0
            msg = "Selected corner frequency is above Nyquist. " + \
                  "Setting Nyquist as high corner."
            warnings.warn(msg)
        while True:
            if order <= maxorder:
                break
            wp = wp * 0.99
            order, wn = cheb2ord(wp, ws, rp, rs, analog=0)
        if ba:
            return cheby2(order, rs, wn, btype='low', analog=0, output='ba')
        z, p, k = cheby2(order, rs, wn, btype='low', analog=0, output='zpk')
        sos = zpk2sos(z, p, k)
        if freq_passband:
            return sosfilt(sos, data, axis=axis), wp * nyquist
        return sosfilt(sos, data, axis=axis)
