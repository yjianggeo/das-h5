"""Convert DAS HDF5 directly to SAC files (streaming, low-memory).

Repackaged from DASh5 das_outsac.py (By Hardy Hong). Algorithms unchanged.
"""

import os
import glob

import numpy as np
import h5py
import pandas as pd
from obspy import UTCDateTime
from tqdm import tqdm

from das_h5.io.writer import save_sac, save_sac_fast
from das_h5.process.resample import decimate

def append_sac(sac_path=None, data=None, ch1=None, ch2=None, npts=None):
    '''
    Append the data to the sac file with Binary file.
    By Hardy Hong, 2024.04.18. Idea from Yuhang Lei and Haoran Song.

    :param sac_path:    The sac file path.
    :param data:        The h5 data needed to append.
    :param ch1:         The data channel beginning.
    :param ch2:         The data channel ending.
    :param npts:        If is not none, changing the sac npts.
    :return:            None
    '''


    for i in range(int(ch1), int(ch2)+1):
        name = '*.' + str(i).zfill(5) + '.*sac'
        sac_name = glob.glob(sac_path + name)
        with open(sac_name[0], "ab") as sf:
            sf.write(data[:, i - ch1].astype('float32'))
            sf.close()
        if npts:
            with open(sac_name[0], "rb+") as sfs:
                content = bytearray(sfs.read())
                npts_wrtite = int(npts).to_bytes(4, 'little')
                content[316:320] = npts_wrtite
                sfs.seek(0)
                sfs.write(content)
                sfs.close()

def h5_to_sac(db_list=None, db_file=None, time_flag=None,
            begin_time=None, end_time=None, channel_begin=None, channel_end=None,
            location_data=None, event_la=None, event_lo=None,  event_depth=None,
            sac_path=None, deci=None, need_sample=None):
    '''
    Read h5 from the time flag what you need.
    By Hardy Hong 2023.06.29

    :param db_list: h5 database list returned from make_dbfile
    :param db_file: h5 database file
    :param time_flag: the time flag that you need
    :param begin_time: relative time beginning from time_flag, it could be negative
    :param end_time: relative end time from time_flag, it could be negative and must > begin_time
    :param channel_begin:  channel window begin
    :param channel_end:   channel_window end
    :param time_flag: the time flag
    :param delta:  data time delta
    :param location_data:  the das location [channel_num, latitude, longitude, altitude]
    :param event_la:  the event latitude
    :param event_lo:  the event longitude
    :param event_depth:  the event depth
    :param sac_path: out sac dir path
    :param deci: data decimate number
    :param need_sample: the sample you need yo apply all data, even data have different sample
    :return: h5_data
    '''

    if db_list is not None:
        db = db_list
    if db_file:
        db = pd.read_csv(db_file, header=None, index_col=None, sep=' ')
        db.columns = ['h5file_name', 'time_label', 'starttime', 'endtime',
                    'delta', 'npts', 'channel_amount']
    time_utc_begin = UTCDateTime(time_flag) + begin_time
    time_utc_end = UTCDateTime(time_flag) + end_time
    channels = channel_end + 1 - channel_begin

    time_index = db[(db['endtime'] >= time_utc_begin) &
            (db['starttime'] <= time_utc_end)].index.tolist()
    db_choose = db.loc[time_index, :]

    # the data of beginning
    try:
        h5 = h5py.File(db_choose['h5file_name'].iloc[0], 'r')
    except Exception as re:
        print('Reading first data... error h5 file number is first file ')
    db_time_start = time_utc_begin - UTCDateTime(db_choose['starttime'].iloc[0])
    db_time_end_final = time_utc_end - UTCDateTime(db_choose['endtime'].iloc[-1])

    dt = db_choose['delta'].iloc[0]
    # if need, giving the input sample. 2024.07.04
    if need_sample:
        dt_new = 1 / int(need_sample)
    else:
        dt_new = dt

    # Determine that you need to fill 0 in the data before it starts
    if db_time_start < 0:
        point_start = 0
        pad0_point_begin = np.zeros([int(abs(db_time_start) / dt), channels], dtype=np.float32)
        if db_time_start <= -0.5:
            print('\nPadding 0 before data for more than '+str(abs(db_time_start))+'s!!!')
    else:
        point_start = int(db_time_start / dt)
    # Determine that you need to fill 0 in the data after it ends
    if db_time_end_final > 0:
        pad0_point_end = np.zeros([int(abs(db_time_end_final) / dt), channels], dtype=np.float32)
        if db_time_end_final >= 0.5:
            print('\nPadding 0 after data for more than '+str(abs(db_time_end_final))+'s!!!')

    if db_choose.shape[0] == 1:
        db_time_end = time_utc_end - UTCDateTime(db_choose['starttime'].iloc[0])
        point_end = int(db_time_end / dt)
        h5_data = h5['Acquisition/Raw[0]/RawData'][point_start:point_end,
                  channel_begin:channel_end + 1]
        if db_time_end_final > 0:
            h5_data = np.concatenate((h5_data, pad0_point_end), axis=0)
        # different sample to decimate 2024.07.04
        factors = int(round(dt_new / dt, 0))
        if factors > 1:
            h5_data = decimate(h5_data, delta=dt, factor=factors)

        # if decimate
        if deci:
            print('\nData decimating ...')
            de_factor = deci
            h5_data = decimate(h5_data, delta=dt_new, factor=de_factor, axis=0)
            dt_new = dt_new * float(de_factor)

        # # convert rad s^-1 to the s^-1   2024.05.15 hht
        # das_gl = h5['Acquisition'].attrs['GaugeLength']
        # h5_data = h5_data * 1.550 * 1e-6 / (4 * np.pi * 1.467 * das_gl * 0.78)
        # # convert rad s^-1 to the s^-1   2025.05.07 jy
        das_gl = h5['Acquisition'].attrs['GaugeLength']
        h5_data = h5_data * 107.9 * 1e-9 / dt / das_gl

        # make the first sac file.
        save_sac(h5_data=h5_data, time_flag=time_flag, delta=dt_new,
            begin_time=begin_time, channel_begin=channel_begin, channel_end=channel_end,
            location_data=location_data, event_la=event_la, event_lo=event_lo,
            event_depth=event_depth, outpath=sac_path)

    if db_choose.shape[0] > 1:
        h5_data = h5['Acquisition/Raw[0]/RawData'][point_start:,
                  channel_begin:channel_end + 1]

        # add padding 0 whether before data.
        if db_time_start < 0:
            h5_data = np.concatenate((pad0_point_begin, h5_data), axis=0)
        # different sample to decimate 2024.07.04
        factors = int(round(dt_new / dt, 0))
        if factors > 1:
            h5_data = decimate(h5_data, delta=dt, factor=factors)

        # if decimate
        if deci:
            print('\nData decimating ...')
            de_factor = deci
            h5_data = decimate(h5_data, delta=dt_new, factor=de_factor, axis=0)
            dt_new = dt_new * float(de_factor)

        # # convert rad s^-1 to the s^-1   2024.05.15 hht
        # das_gl = h5['Acquisition'].attrs['GaugeLength']
        # h5_data = h5_data * 1.550 * 1e-6 / (4 * np.pi * 1.467 * das_gl * 0.78)
        # # convert rad s^-1 to the s^-1   2025.05.07 jy
        das_gl = h5['Acquisition'].attrs['GaugeLength']
        h5_data = h5_data * 107.9 * 1e-9 / dt / das_gl

        # make the first sac file.
        save_sac_fast(h5_data=h5_data, time_flag=time_flag, delta=dt_new,
            begin_time=begin_time, channel_begin=channel_begin, channel_end=channel_end,
            location_data=location_data, event_la=event_la, event_lo=event_lo,
            event_depth=event_depth, outpath=sac_path)

        npts_cal = len(h5_data.copy())
        time_gap_all = 0
        # for i in range(1, len(time_index)):
        print('>>>···<<<')
        for i in tqdm(range(1, len(time_index)), desc='Getting h5 data and converting to sac process: '):  # hht 2024.04.17
            h5_data = np.array([], dtype=np.float32)
            # # Check whether the memory is full
            # per = psutil.virtual_memory().percent
            # if per >= 95:
            #     print('\nWarning! The memory percent is over 95%!')
            #     print('\nNow data range...')
            #     print('>>>')
            #     print(time_utc_begin.strftime("%Y-%m-%dT%H:%M:%S"))
            #     print('···')
            #     per_end_time = UTCDateTime(db_choose['endtime'].iloc[i-1])
            #     print(per_end_time.strftime("%Y-%m-%dT%H:%M:%S"))
            #     print('<<<')
            #     break

            # whether exist data gap, if yes, fill with zeros
            dt = db_choose['delta'].iloc[i]
            time_gap_flag = UTCDateTime(db_choose['starttime'].iloc[i]) - \
                            UTCDateTime(db_choose['endtime'].iloc[i-1])
            time_gap_all += time_gap_flag
            gap_npts = int(time_gap_flag / dt)

            # if gap_npts >= 1:
            #     fill_values = np.zeros([gap_npts, channels], dtype=np.float32)
            #     h5_data = fill_values
            #     # h5_data = np.concatenate((h5_data, fill_values), axis=0)

            try:
                h5 = h5py.File(db_choose['h5file_name'].iloc[i], 'r')
            except Exception as re:
                print('\nReading data... error h5 file number is ' + db_choose['h5file_name'].iloc[i])

            if i == len(time_index) - 1:   # only 2 file or the last file
                db_time_end = time_utc_end - UTCDateTime(db_choose['starttime'].iloc[i])
                point_end = int(db_time_end / dt)
                if gap_npts < 0:
                    h5_data = h5['Acquisition/Raw[0]/RawData'][:point_end+gap_npts, channel_begin:channel_end + 1]
                else:
                    h5_data = h5['Acquisition/Raw[0]/RawData'][:point_end, channel_begin:channel_end + 1]
            else:
                if gap_npts < 0:
                    h5_data = h5['Acquisition/Raw[0]/RawData'][:gap_npts, channel_begin:channel_end + 1]
                else:
                    h5_data = h5['Acquisition/Raw[0]/RawData'][:, channel_begin:channel_end + 1]

            if gap_npts >= 1:
                fill_values = np.zeros([gap_npts, channels], dtype=np.float32)
                h5_data = np.concatenate((fill_values, h5_data), axis=0)

            # when the final h5 data appended, then changing the npts
            if i == len(time_index)-1:
                # add padding 0 whether after data.
                if db_time_end_final > 0:
                    h5_data = np.concatenate((h5_data, pad0_point_end), axis=0)

                # different sample to decimate 2024.07.04
                factors = int(round(dt_new / dt, 0))
                if factors > 1:
                    h5_data = decimate(h5_data, delta=dt, factor=factors)
                # if decimate
                if deci:
                    de_factor = deci
                    h5_data = decimate(h5_data, delta=dt_new, factor=de_factor, axis=0)
                npts_cal += len(h5_data)

                # # convert rad s^-1 to the s^-1   2024.05.15 hht
                # das_gl = h5['Acquisition'].attrs['GaugeLength']
                # h5_data = h5_data * 1.550 * 1e-6 / (4 * np.pi * 1.467 * das_gl * 0.78)
                # # convert rad s^-1 to the s^-1   2025.05.07 jy
                das_gl = h5['Acquisition'].attrs['GaugeLength']
                h5_data = h5_data * 107.9 * 1e-9 / dt / das_gl

                # append the sac data of binary
                append_sac(sac_path=sac_path, data=h5_data, ch1=channel_begin, ch2=channel_end, npts=npts_cal)

            else:
                # different sample to decimate 2024.07.04
                factors = int(round(dt_new / dt, 0))
                if factors > 1:
                    h5_data = decimate(h5_data, delta=dt, factor=factors)
                # if decimate
                if deci:
                    de_factor = deci
                    h5_data = decimate(h5_data, delta=dt_new, factor=de_factor, axis=0)

                # # convert rad s^-1 to the s^-1   2024.05.15 hht
                # das_gl = h5['Acquisition'].attrs['GaugeLength']
                # h5_data = h5_data * 1.550 * 1e-6 / (4 * np.pi * 1.467 * das_gl * 0.78)
                # # convert rad s^-1 to the s^-1   2025.05.07 jy
                das_gl = h5['Acquisition'].attrs['GaugeLength']
                h5_data = h5_data * 107.9 * 1e-9 / dt / das_gl

                # append the sac data of binary
                append_sac(sac_path=sac_path, data=h5_data, ch1=channel_begin, ch2=channel_end)
                npts_cal += len(h5_data)
            # delete h5 values
            del h5

        if time_gap_all > (db_choose.shape[0] - 1) * 0.05:
            print('\nPadding 0 in data for more than '+str(abs(time_gap_all))+'s!!!')

    print('\nConverting H5 to sac successfully!')

    return dt
