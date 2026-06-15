"""DAS resampling: anti-alias decimation and interpolation.

Repackaged from DASh5 das_processing.py (By Hardy Hong). Algorithms unchanged.
"""

import numpy as np

from das_h5.process.filters import filter

def decimate(data, delta=0.01, factor=1, filter_need=True, axis=0):
    '''
    To decimate the data array.
    By Hardy Hong 2023.07.11

    :param data: data array
    :param n:    decimate scale
    :return:     data array
    '''

    df = 1 / delta
    # do automatic lowpass filtering
    if filter_need:
        # be sure filter still behaves good
        if factor > 16:
            msg = "Automatic filter design is unstable for decimation " + \
                  "factors above 16. Manual decimation is necessary."
            raise ArithmeticError(msg)
        freq = df * 0.5 / float(factor)
        data = filter(data, delta=delta, type='lowpass_cheby_2', freq=freq, maxorder=16, axis=axis)

    if not isinstance(factor, int):
        msg = "Decimation_factor must be an integer!"
        raise TypeError(msg)

    # reshape and only use every decimation_factor-th sample
    data = np.array(data[::factor, :])
    return data

def interpolate(data, delta=0.01, factor=1, axis=0, type='linear'):
    '''
    Interpolate the h5 data [array]
    By Hardy Hong 2023.07.11

    :param data:    data array
    :param delta:   old delta
    :param factor:  interp factor
    :param axis:    which axis to interp
    :param type:    which type to interp
    :return:        data array  with new-delta = delta / factor
    '''

    from scipy.interpolate import interp1d
    time_len = data.shape[0]
    time = np.arange(0, time_len * delta, delta)
    time_new = np.arange(0, time_len * delta, delta / factor)
    f = interp1d(time, data, kind=type, axis=axis)
    return f(time_new)
