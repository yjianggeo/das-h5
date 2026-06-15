"""Write DAS data to SAC and NumPy formats.

Repackaged from DASh5 das_writedata.py (By Hardy Hong). Algorithms unchanged.
"""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

def save_sac(h5_data, time_flag='obspy_utctime', delta=0.001,
            begin_time=0, channel_begin=0, channel_end=30,
            location_data=None, event_la=None, event_lo=None,  event_depth=None,
             outpath='sac_out_dir'):
    '''
    Whether to save sac data.
    By Hardy Hong 2023.06.30

    :param h5_data: the complete numpy raw data [channel_begin, channel_end] || include channel_end
    :param time_flag: the time flag
    :param delta:  data time delta
    :param begin_time: relative time beginning from time_flag, it could be negative
    :param channel_begin:  channel window begin
    :param channel_end:   channel_window end
    :param location_data:  the das location [channel_num, latitude, longitude, altitude]
    :param event_la:  the event latitude
    :param event_lo:  the event longitude
    :param event_depth:  the event depth
    :return:  'Save sac data successfully!'
    '''

    from obspy import UTCDateTime, Stream, Trace, read
    from obspy.io.sac.sactrace import SACTrace


    #das station locations
    if location_data:
        try:
            location = pd.read_csv(location_data, names=['Channel_number', 'Latitude',
                                        'Longitude', 'Altitude'], header=None, sep=" ")

        except:
            try:
                location = pd.read_csv(location_data, names=['Channel_number', 'Latitude',
                                                             'Longitude', 'Altitude'], header=None, sep=",")
            except:
                print('Error! Loc file delimiter is neither " " nor "," !!! ')
        if location.all:
            print('\nGet DAS locations ...')
        else:
            print('\nError! Can not get DAS locations!')

    starttime = UTCDateTime(time_flag)
    # for i in range(channel_begin, channel_end + 1):
    for i in tqdm(range(channel_begin, channel_end + 1), desc='Saving Sac data process: '):  # hht 2024.04.09
        stats_m = {'delta': delta, 'starttime': starttime,
                   'station': str(i).zfill(5), 'network': 'DAS'}
        tr_m = Trace(data=h5_data[:, i - channel_begin], header=stats_m)
        name = f"Das.{str(i).zfill(5)}.{starttime.strftime('%Y%m%d%H%M%S')}.sac"
        sac = SACTrace.from_obspy_trace(tr_m, keep_sac_header=False)
        sac.b = begin_time
        if location_data:
            channel_meta = location.loc[location['Channel_number'] == int(sac.kstnm)]
            if not(channel_meta.empty):
                sac.stla = float(channel_meta['Latitude'])
                sac.stlo = float(channel_meta['Longitude'])
                # sac.stdp = float(channel_meta['Altitude']) / 1000
                sac.stdp = float(0)
        sac.lcalda = True
        sac.o = 0
        sac.iztype = 'io'
        if event_la and event_lo:  # event location
            sac.evla = float(event_la)
            sac.evlo = float(event_lo)
            sac.evdp = float(event_depth)

        if not os.path.exists(outpath):
            os.mkdir(outpath)
        sac.write(os.path.join(outpath, name), byteorder='little')
    return str('Saving sac data successfully!')

def save_sac_fast(h5_data, time_flag='obspy_utctime', delta=0.001,
            begin_time=0, channel_begin=0, channel_end=30,
            location_data=None, event_la=None, event_lo=None,  event_depth=None,
             outpath='sac_out_dir'):
    '''
    Whether to save sac data.
    By Hardy Hong 2023.06.30

    :param h5_data: the complete numpy raw data [channel_begin, channel_end] || include channel_end
    :param time_flag: the time flag
    :param delta:  data time delta
    :param begin_time: relative time beginning from time_flag, it could be negative
    :param channel_begin:  channel window begin
    :param channel_end:   channel_window end
    :param location_data:  the das location [channel_num, latitude, longitude, altitude]
    :param event_la:  the event latitude
    :param event_lo:  the event longitude
    :param event_depth:  the event depth
    :return:  'Save sac data successfully!'
    '''

    from obspy import UTCDateTime, Stream, Trace, read
    from obspy.io.sac.sactrace import SACTrace


    #das station locations
    if location_data:
        try:
            location = pd.read_csv(location_data, names=['Channel_number', 'Latitude',
                                        'Longitude', 'Altitude'], header=None, sep=" ")

        except:
            try:
                location = pd.read_csv(location_data, names=['Channel_number', 'Latitude',
                                                             'Longitude', 'Altitude'], header=None, sep=",")
            except:
                print('Error! Loc file delimiter is neither " " nor "," !!! ')
        if location.all:
            print('\nGet DAS locations ...')
        else:
            print('\nError! Can not get DAS locations!')

    starttime = UTCDateTime(time_flag)
    # for i in range(channel_begin, channel_end + 1):
    for i in range(channel_begin, channel_end + 1):  # hht 2024.04.09
        stats_m = {'delta': delta, 'starttime': starttime,
                   'station': str(i).zfill(5), 'network': 'DAS'}
        tr_m = Trace(data=h5_data[:, i - channel_begin], header=stats_m)
        name = f"Das.{str(i).zfill(5)}.{starttime.strftime('%Y%m%d%H%M%S')}.sac"
        sac = SACTrace.from_obspy_trace(tr_m, keep_sac_header=False)
        sac.b = begin_time
        if location_data:
            channel_meta = location.loc[location['Channel_number'] == int(sac.kstnm)]
            if not(channel_meta.empty):
                sac.stla = float(channel_meta['Latitude'])
                sac.stlo = float(channel_meta['Longitude'])
                # sac.stdp = float(channel_meta['Altitude']) / 1000
                sac.stdp = float(0)
        sac.lcalda = True
        sac.o = 0
        sac.iztype = 'io'
        if event_la and event_lo:  # event location
            sac.evla = float(event_la)
            sac.evlo = float(event_lo)
            sac.evdp = float(event_depth)

        if not os.path.exists(outpath):
            os.mkdir(outpath)
        sac.write(os.path.join(outpath, name), byteorder='little')
    print('\nSaving first sac data successfully!')
    return str('Saving first sac data successfully!')

def save_npz(data, outnpy=None):
    '''
    Save h5 data to numpy array.
    By Hardy Hong 2023.07.13

    (npy name must be Das.origin_time.sample_rate.channel_begin.channel_end.npy
                                  e.g. Das.SP002.20230615131420.01000.00000.02000.npy  (SP starttime is positive +)
                                       Das.SN002.20230615131420.01000.00000.02000.npy  (SN starttime is negative -))

    '''

    np.save(outnpy, data)
    return str('Save npy data successfully!')
