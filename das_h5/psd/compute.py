"""Compute Welch PSD spectra and derived statistics for DAS data.

Repackaged from DASh5 das_psd.py (By Hardy Hong). Algorithms unchanged.
"""

import os
import pickle

import numpy as np
from obspy import read, Stream, Trace, UTCDateTime

from das_h5.process.filters import nextpow2


def das_psd(das_data, delta, wind_len, wind_step, minfreq, maxfreq, ch1,
            starttime=None, nfft=None, scaling='density', out_psd=None):
    '''
    Calculate the psd or ps from the das data array.
    By Hardy Hong 2024.01.15

    :param das_data:        das data array [times, channels]
    :param delta:           time delta
    :param wind_len:        segment length [s]
    :param wind_step:       moving time step [s]
    :param minfreq:         min freq or min freq matrix
    :param maxfreq:         max freq or max freq matrix
    :param ch1:             channel beginning
    :param starttime:       das data starttime [UTCDateTime]
    :param nfft:            the points of calculating fft
    :param scaling:         'density' [V**2/Hz] or 'sepctrum' [V**2], default is 'density'
    :param out_psd:          output path the psd data .pkl
    :return: [list]          freq series, and psd values, times series [UTCDateTime], max value of every channel.
    '''
    from scipy.signal import welch

    times = []  # time series [UTCDateTime]
    if nfft:
        nfft = nextpow2(nfft)
    else:
        nfft = nextpow2(int(wind_len / delta))
    nsamp = int(wind_len / delta)
    nstep = int(wind_step / delta)
    deltaf = 1 / delta / nfft
    nwin = int(np.array(das_data.shape[0] / nsamp)) * int(wind_len / wind_step) - 1
    fss_all = []
    psds_all = []

    if type(minfreq) is int:
        nlow = int(minfreq / float(deltaf) + 0.5)
        nhigh = int(maxfreq / float(deltaf) + 0.5)
        nlow = max(1, nlow)  # avoid using the offset
        nhigh = min(nfft // 2 - 1, nhigh)  # avoid using nyquist
    else:
        minfreq = np.asarray(minfreq)
        maxfreq = np.asarray(maxfreq)
        nlow = np.array(minfreq / deltaf + 0.5, dtype=int)
        nhigh = np.array(maxfreq / deltaf + 0.5, dtype=int)
        for i in range(len(nlow)):
            nlow[i] = max(1, nlow[i])  # avoid using the offset
            nhigh[i] = min(nfft // 2 - 1, nhigh[i])  # avoid using nyquist

    knum = 0
    max_psds = np.zeros(das_data.shape[1])
    print('>>>·<<<')
    print('From ' + starttime.strftime('%Y-%m-%dT%H:%M:%S'))
    print('···')
    for i in range(nwin):
        if i * nstep + nsamp < np.array(das_data.shape[0]):
            # print(UTCDateTime(starttime+i*wind_step))      ##test where is wrong in cal the psd.
            times.append(UTCDateTime(starttime + i * wind_step))
            data_seg = das_data[i * nstep:i * nstep + nsamp, :]
            fss, psds = welch(data_seg, fs=1 / delta, nperseg=data_seg.shape[0], detrend='constant', window='hann',
                              nfft=nfft, return_onesided=True, scaling=scaling, axis=0, average='mean', noverlap=0)

            max_pp = np.max(psds, axis=0)
            for pp in range(len(max_psds)):
                if max_pp[pp] > max_psds[pp]:
                    max_psds[pp] = max_pp[pp]

            if type(minfreq) is int:
                fss_all.append(fss[nlow:nhigh + 1])

                psds_all.append(psds[nlow:nhigh + 1, :])
            else:
                fss_matrix = []
                psds_matrix = []
                for j in range(len(minfreq)):
                    fss_matrix.append(fss[nlow[j]:nhigh[j] + 1])
                    psds_matrix.append(psds[nlow[j]:nhigh[j] + 1, :])
                fss_all.append(fss_matrix)
                psds_all.append(psds_matrix)
            knum += 1
    print('To   ' + times[-1].strftime('%Y-%m-%dT%H:%M:%S'))
    print('>>>·<<<')
    print('Total ' + str(knum) + ' segments...')

    # output data and db file
    if out_psd:
        print('\nSaving PSD-pkl... ')
        if not os.path.exists(out_psd):
            os.mkdir(out_psd)
        ch2 = ch1 + das_data.shape[1] - 1
        psds_paras = [wind_len, wind_step, ch1, ch2, minfreq, maxfreq]
        data_all_save = [psds_paras, fss_all, psds_all, times, max_psds]
        pkl_name = 'PSD.' + str(ch1).zfill(5) + '.' + str(ch2).zfill(5) + '.' + times[0].strftime(
            '%Y%m%d%H%M%S') + '.pkl'
        try:
            with open(out_psd + pkl_name, 'wb') as f:
                pickle.dump(data_all_save, f)
        except:
            print('\nSaving PSD-pkl is wrong!')
        print('\nSaving PSD-pkl >>> ' + out_psd + pkl_name + ' Successfully!')

    return fss_all, psds_all, times, max_psds


def das_psd_type(fss, psds, times, freq_band, num, ch1, ch_choose, bins):
    '''
    Get different values of plotting input.
    By Hardy Hong 2024.03.12

    :param fss:         freq form func of das_psd
    :param psds:        psd values form func of das_psd
    :param times:       times from func of das_psd
    :param freq_band    the amount of freq bands
    :param num:         which freq band to calculate
    :param ch1:         begin channel of input data
    :param ch_choose:   which channel you need calculate
    :param bins:        for statistic fig to interp data
    :param wind_len:        segment length [s]
    :param wind_step:       moving time step [s]
    :param out_pdb      the path of out psd data base file
    :return: psds_rms, psds_freq_all, psds_singel_time, psds_singel_time_freq_band,
            psds_single_band, psds_single_time_psds, psds_single_time_freqs
    '''

    # for psd rms with different time.
    psds_rms = np.zeros((len(times), psds[0][num].shape[1]))
    # for psd sum with different freq in a time range.
    psds_freq_all = np.zeros((psds[0][num].shape[0], psds[0][num].shape[1]))
    # for psd of single channel with different time.
    psds_singel_time = np.zeros((psds[0][num].shape[0], len(times)))
    # for mean psd of single channel of different time with different freq.
    psds_singel_time_freq_band = np.zeros((len(times), freq_band))
    for i in range(len(times)):
        # if data_norm == "raw-self" or data_norm == "both":
        #     psds[i] = psds[i] / max(max_psds)
        nfreq = psds[i][num].shape[0]
        psds_seg = psds[i][num].copy()
        psds_seg = np.sum(psds_seg, axis=0)
        psds_seg = np.sqrt(psds_seg)
        # psds_seg = np.sum(psds[i], axis=0) / nfreq
        psds_rms[i, :] = psds_seg
        psds_freq_all += psds[i][num].copy()
        psds_singel_time[:, i] = psds[i][num][:, ch_choose - ch1].copy()

        for j in range(freq_band):
            psds_channel = psds[i][j][:, ch_choose - ch1].copy()
            psds_channel_mean = np.mean(psds_channel)
            psds_singel_time_freq_band[i, j] = psds_channel_mean

    # for mean psd of single channel of different time with different freq.  10*log10  [db]
    psds_single_band = 10 * np.log10(psds_singel_time_freq_band)

    # for statistic psd
    psds_single_time_freqs = np.array([])
    psds_single_time_psds = np.array([])
    if bins > len(fss[0][num]):
        fss_new = np.linspace(fss[0][num][0], fss[0][num][-1], bins)
        for i in range(len(times)):
            psds_new = np.interp(fss_new, fss[i][num], psds[i][num][:, ch_choose - ch1])
            psds_single_time_freqs = np.hstack((psds_single_time_freqs, fss_new))
            psds_single_time_psds = np.hstack((psds_single_time_psds, psds_new))
    else:
        for i in range(len(times)):
            psds_single_time_freqs = np.hstack((psds_single_time_freqs, fss[i][num]))
            psds_single_time_psds = np.hstack((psds_single_time_psds, psds[i][num][:, ch_choose - ch1]))
    psds_single_time_psds = 10 * np.log10(psds_single_time_psds)

    return psds_rms, psds_freq_all, psds_singel_time, psds_singel_time_freq_band, \
           psds_single_band, psds_single_time_psds, psds_single_time_freqs


def das_psd_pkl(pkl_path, ch1, ch2):
    '''
    Read the psd pkl data, and output the needed data.
    By Hardy Hong 2024.03.20

    :param pkl_path:    pkl data path.
    :param ch1:         channel beginning.
    :param ch2:         channel ending
    :return:            fss_all, psds_all, times, max_psds (just like das_psd func)
    '''

    pkl_list = os.listdir(pkl_path)
    if len(pkl_list) == 0:
        print('Error! There is not any *.pkl data!')
        exit()
    pkl_list.sort()
    pkl_all = []
    chb = []
    che = []
    choose_num = []

    for i in range(len(pkl_list)):
        chb.append(int(pkl_list[i].split('.')[1]))
        che.append(int(pkl_list[i].split('.')[2]))
        if (chb[i] <= ch1 <= che[i]) or (chb[i] <= ch2 <= che[i]):
            choose_num.append(i)

    if not choose_num:
        num_b = 0
        num_e = len(chb)
    else:
        num_b = choose_num[0]
        num_e = choose_num[-1]

    with open(pkl_path + pkl_list[num_b], 'rb') as f:
        a = pickle.load(f)
        psds_first = a[2]
        fss_first = a[1]
        time_first = a[3]
        psdmax_first = a[4]

    # padding beginning
    if chb[num_b] > ch1:
        print('\nPadding 0 of beginning channels ' + str(ch1) + '-' + str(chb[0] - 1))
        for j in range(len(a[2])):
            for k in range(len(a[2][0])):
                pad0 = np.zeros([a[2][j][k].shape[0], (ch1 - chb[num_b])], dtype=np.float32)
                psds_first[j][k] = np.concatenate((pad0, psds_first[j][k]), axis=1)
    else:
        for j in range(len(a[2])):
            for k in range(len(a[2][0])):
                psds_first[j][k] = a[2][j][k][:, (ch1 - chb[num_b]):]

    for i in range(num_b, num_e):
        if i > num_b:
            with open(pkl_path + pkl_list[i], 'rb') as f:
                a = pickle.load(f)
                pkl_all.append(a)
            # padding 0
            if chb[i] > che[i - 1] + 1:
                for j in range(len(a[2])):
                    for k in range(len(a[2][0])):
                        pad0 = np.zeros([a[2][j][k].shape[0], (chb[i] - che[i - 1])], dtype=np.float32)
                        psds_first[j][k] = np.concatenate((psds_first[j][k], pad0), axis=1)
                        psds_first[j][k] = np.concatenate((psds_first[j][k], a[2][j][k]), axis=1)
            elif chb[i] < che[i - 1] + 1:
                for j in range(len(a[2])):
                    for k in range(len(a[2][0])):
                        psds_first[j][k] = np.concatenate((psds_first[j][k], a[2][j][k][:, (che[i - 1] - chb[i]):]),
                                                          axis=1)
            else:
                for j in range(len(a[2])):
                    for k in range(len(a[2][0])):
                        psds_first[j][k] = np.concatenate((psds_first[j][k], a[2][j][k]), axis=1)

    # padding ending
    if che[num_e] < ch2:
        print('\nPadding 0 of ending channels ' + str(che[num_e] + 1) + '-' + str(ch2))
        for j in range(len(a[2])):
            for k in range(len(a[2][0])):
                pad0 = np.zeros([a[2][j][k].shape[0], (ch2 - che[num_e])], dtype=np.float32)
                psds_first[j][k] = np.concatenate((psds_first[j][k], pad0), axis=1)
    elif che[num_e] > ch2:
        for j in range(len(a[2])):
            for k in range(len(a[2][0])):
                psds_first[j][k] = psds_first[j][k][:(che[num_e] - ch2)]
    else:
        for j in range(len(a[2])):
            for k in range(len(a[2][0])):
                psds_first[j][k] = psds_first[j][k]

    return fss_first, psds_first, time_first, psdmax_first
