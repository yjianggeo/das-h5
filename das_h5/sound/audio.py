"""Convert a DAS trace to an audible .wav file.

Repackaged from DASh5 das_sound.py (By Hardy Hong). Algorithm unchanged.
"""

import os
import wave

import numpy as np
import scipy.signal as signal


def das_sound(data, delta, title, scale=1.0, gain=5000, save_path=None, interp=None):
    '''
    Convert das data to sound (.wav)
    By Hardy Hong 2024.03.29
    :param data:   das single channel data.
    :param delta:  data delta (1/sample)
    :param title:  the sound title
    :param scale:  delta scale  >>> sample =  int(1/delta) * scale
    :param save_path:   whether to save sound (save path)
    :param interp: whether to interp data.
    :return:  None
    '''

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # timex = np.arange(0, len(data), 1)
    if interp:
        data = signal.resample_poly(data, interp, 1)
    framerate = int(1/delta*scale)
    data = data * gain
    data = data.astype(np.short)
    # print(str(framerate) + '.' + str(len(data)))
    f = wave.open(save_path+title + '.wav', 'wb')
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(framerate)
    if save_path:
        f.writeframes(data)
    f.close()
