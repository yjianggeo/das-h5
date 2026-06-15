"""Read DAS HDF5 files, build the database index, and extract data windows.

Repackaged from DASh5 das_readdata.py (By Hardy Hong). Algorithms unchanged.
"""

import os
from glob import glob

import numpy as np
import h5py
import pandas as pd
import psutil
from obspy import UTCDateTime
from tqdm import tqdm, trange

from das_h5.process.resample import decimate


def get_database_h5(h5_listname):
    '''
    Read the h5 files, and return data what you need (Here is dbfile).
    By Hardy Hong 2023.06.28

    :param h5_listname: The list of h5 file.
    :return: db_list[npts, delta, starttime, endtime, trace amount]
    '''

    npts = []
    delta = []
    starttime_m = []
    endtime_m = []
    channel_amount = []
    time_label = []
    file_path = []
    db_list = [file_path, time_label, starttime_m, endtime_m, delta, npts, channel_amount]
    for i in range(len(h5_listname)):
        try:
            h5 = h5py.File(h5_listname[i], 'r')
            # datam = h5['Acquisition/Raw[0]/RawData']
            file_path = os.path.abspath(h5_listname[i])
            npts = h5['Acquisition/Raw[0]/RawData'].shape[0]
            channel_amount = h5['Acquisition/Raw[0]/RawData'].shape[1]
            delta = 1 / h5['Acquisition/Raw[0]'].attrs['OutputDataRate']
            t = h5['Acquisition/Raw[0]/RawDataTime'][0]
            starttime_value = UTCDateTime(UTCDateTime(0) + int(t / 1e6) + t % 1e6 / 1e6)
            endtime_value = UTCDateTime(UTCDateTime(0) + int(t / 1e6) + t % 1e6 / 1e6 + npts * delta)
            starttime_m = starttime_value.strftime('%Y-%m-%dT%H:%M:%S.%f')
            endtime_m = endtime_value.strftime('%Y-%m-%dT%H:%M:%S.%f')
            time_label = starttime_value.strftime('%m%d%H%M')
            db_list[0].append(file_path)
            db_list[5].append(npts)
            db_list[6].append(channel_amount)
            db_list[4].append(delta)
            db_list[2].append(starttime_m)
            db_list[3].append(endtime_m)
            db_list[1].append(time_label)
        except Exception as e:
            pass
        continue
    return db_list

def make_dbfile(inpath=None, outfile=None):
    '''
    The function to make database file.
    By Hardy Hong 2023.06.29 || Modified on 2024.05.16

    :param inpath:  h5 file path
    :param outfile: h5 file db txt [h5file_name time_label
                        starttime endtime delta npts channel_amount]
    :return: db_list_all: it should be saved firstly (pandas type)
    '''
    db_list1 = [[], [], [], [], [], [], []]
    for dir_path, dir_names, file_names in os.walk(inpath):
        h5file_list = [c for c in glob(dir_path+'/*h5')]
        if h5file_list:
            db_list0 = get_database_h5(h5file_list)
            for dbi in range(len(db_list1)):
                db_list1[dbi].extend(db_list0[dbi])

    db_list_all = pd.DataFrame(data={
        'h5file_name': db_list1[0],
        'time_label': db_list1[1],
        'starttime': db_list1[2],
        'endtime': db_list1[3],
        'delta': db_list1[4],
        'npts': db_list1[5],
        'channel_amount': db_list1[6]
    })
    db_list_all = db_list_all.sort_values(by="time_label", ascending=True)
    if outfile and not db_list_all.empty:
        try:
            db_list_all.to_csv(outfile, sep=' ', header=False, index=False)
        except Exception as re:
            print('\nSaving db_file is wrong!')
        print('\nSaving db_file -- ', outfile, 'successfully!')
    elif db_list_all.empty:
        print('\nGetting nothing! Saving db_file is wrong!')


    return db_list_all

def read_h5(db_list=None, db_file=None, time_flag=None, need_sample=None,
            begin_time=0, end_time=10, channel_begin=0, channel_end=30):
    '''
    Read h5 from the time flag what you need.
    By Hardy Hong 2023.06.29

    :param db_list: h5 database list returned from make_dbfile
    :param db_file: h5 database file
    :param time_flag: the time flag that you need
    :param need_sample: the sample you need yo apply all data, even data have different sample.
    :param begin_time: relative time beginning from time_flag, it could be negative
    :param end_time: relative end time from time_flag, it could be negative and must > begin_time
    :param channel_begin:  channel window begin
    :param channel_end:   channel_window end
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
    # if need, giving the input sample
    if need_sample:
        dt_new = 1 / int(need_sample)
    else:
        dt_new = dt


    # Determine that you need to fill 0 in the data before it starts
    if db_time_start < 0:
        point_start = 0
        pad0_point_begin = np.zeros([int(abs(db_time_start) / dt_new), channels], dtype=np.float32)
        if db_time_start <= -0.5:
            print('\nPadding 0 before data for more than '+str(abs(db_time_start))+'s!!!')
    else:
        point_start = int(db_time_start / dt)
    # Determine that you need to fill 0 in the data after it ends
    if db_time_end_final > 0:
        pad0_point_end = np.zeros([int(abs(db_time_end_final) / dt_new), channels], dtype=np.float32)
        if db_time_end_final >= 0.5:
            print('\nPadding 0 after data for more than '+str(abs(db_time_end_final))+'s!!!')

    if db_choose.shape[0] == 1:
        db_time_end = time_utc_end - UTCDateTime(db_choose['starttime'].iloc[0])
        point_end = int(db_time_end / dt)
        h5_data = h5['Acquisition/Raw[0]/RawData'][point_start:point_end,
                  channel_begin:channel_end + 1]
        # different sample to decimate 2024.07.04
        factors = int(round(dt_new / dt, 0))
        if factors > 1:
            h5_data = decimate(h5_data, delta=dt, factor=factors)
        # get gauge length 2024.05.15
        das_gl = h5['Acquisition'].attrs['GaugeLength']

    if db_choose.shape[0] > 1:
        h5_data = h5['Acquisition/Raw[0]/RawData'][point_start:,
                  channel_begin:channel_end + 1]
        # different sample to decimate 2024.07.04
        factors = int(round(dt_new / dt, 0))
        if factors > 1:
            h5_data = decimate(h5_data, delta=dt, factor=factors)
        # get gauge length 2024.05.15
        das_gl = h5['Acquisition'].attrs['GaugeLength']
        time_gap_all = 0
        # for i in range(1, len(time_index)):
        for i in tqdm(range(1, len(time_index)), desc='Getting h5 data process: '):  # hht 2024.04.17
            # Check whether the memory is full
            per = psutil.virtual_memory().percent
            if per >= 95:
                print('\nWarning! The memory percent is over 95%!')
                print('\nNow data range...')
                print('>>>')
                print(time_utc_begin.strftime("%Y-%m-%dT%H:%M:%S"))
                print('···')
                per_end_time = UTCDateTime(db_choose['endtime'].iloc[i-1])
                print(per_end_time.strftime("%Y-%m-%dT%H:%M:%S"))
                print('<<<')
                break
            # whether exist data gap, if yes, fill with zeros
            time_gap_flag = UTCDateTime(db_choose['starttime'].iloc[i]) - \
                            UTCDateTime(db_choose['endtime'].iloc[i-1])
            time_gap_all += time_gap_flag
            gap_npts = int(time_gap_flag / dt_new)
            gap_npts_old = int(time_gap_flag / dt)

            if gap_npts >= 1:
                fill_values = np.zeros([gap_npts, channels], dtype=np.float32)
                h5_data = np.concatenate((h5_data, fill_values), axis=0)
            try:
                h5 = h5py.File(db_choose['h5file_name'].iloc[i], 'r')
            except Exception as re:
                print('\nReading data... error h5 file number is ' + db_choose['h5file_name'].iloc[i])
            dt = db_choose['delta'].iloc[i]
            if i == len(time_index) - 1:   # only 2 file or the last file
                db_time_end = time_utc_end - UTCDateTime(db_choose['starttime'].iloc[i])
                point_end = int(db_time_end / dt)
                if gap_npts_old < 0:
                    h5_data_new = h5['Acquisition/Raw[0]/RawData'][:point_end + gap_npts_old,
                                  channel_begin:channel_end + 1]
                else:
                    h5_data_new = h5['Acquisition/Raw[0]/RawData'][:point_end, channel_begin:channel_end + 1]

                # different sample to decimate 2024.07.04
                factors = int(round(dt_new / dt, 0))
                if factors > 1:
                    h5_data_new = decimate(h5_data_new, delta=dt, factor=factors)
                h5_data = np.concatenate((h5_data, h5_data_new), axis=0)

            else:
                if gap_npts_old < 0:
                    h5_data_new = h5['Acquisition/Raw[0]/RawData'][:gap_npts_old, channel_begin:channel_end + 1]
                else:
                    h5_data_new = h5['Acquisition/Raw[0]/RawData'][:, channel_begin:channel_end + 1]

                # different sample to decimate 2024.07.04
                factors = int(round(dt_new / dt, 0))
                if factors > 1:
                    h5_data_new = decimate(h5_data_new, delta=dt, factor=factors)
                h5_data = np.concatenate((h5_data, h5_data_new), axis=0)
            # delete h5 values
            del h5

        if time_gap_all > (db_choose.shape[0] - 1) * 0.05:
            print('\nPadding 0 in data for more than '+str(abs(time_gap_all))+'s!!!')

    # add padding 0 whether before or after data.
    if db_time_start < 0:
        h5_data = np.concatenate((pad0_point_begin, h5_data), axis=0)
    if db_time_end_final > 0:
        h5_data = np.concatenate((h5_data, pad0_point_end), axis=0)

    # # convert rad s^-1 to the s^-1   2024.05.15 hht
    h5_data = h5_data * 1.550 * 1e-6 / (4 * np.pi * 1.467 * das_gl * 0.78)
    return h5_data, dt

def read_sac(sacpath=None, ref_time=None, begin_time=0, end_time=10,
             begin_channel=0, end_channel=10):
    '''
    Convert sac data to numpy array.
    By Hardy Hong 2023.07.13

    :param sacpath:         input sac dir path
    :param ref_time:        reference time
    :param begin_time:      relative begin time what you need
    :param end_time:        relative end time what you need
    :param begin_channel:   channel start what you need
    :param end_channel:     channel end what you need
    :return:                data array, delta
    '''

    from obspy import read, UTCDateTime, Stream

    sac_list = os.listdir(sacpath)
    sac_list.sort()
    st_example = read(sacpath + sac_list[0])
    delta = st_example[0].stats.delta
    if UTCDateTime(ref_time) + begin_time < st_example[0].stats.starttime:
        print('Begin time must be in range of data!')
    elif UTCDateTime(ref_time) + end_time - 3 * delta > st_example[0].stats.endtime:
        print('End time must be in range of data!')

    channel_sac_flag = []
    channel_sac_num = []
    for i in range(len(sac_list)):
        channel_sac = int(sac_list[i].split('.')[1])
        if channel_sac >= begin_channel and channel_sac <= end_channel:
            channel_sac_flag.append(i)
            channel_sac_num.append(channel_sac)

    st = read(sacpath + sac_list[int(channel_sac_flag[0])], starttime=UTCDateTime(ref_time) + begin_time,
                  endtime=UTCDateTime(ref_time) + end_time)
    data = st[0].data
    data = data.T

    for k in range(1, len(channel_sac_num)):
        st = read(sacpath + sac_list[int(channel_sac_flag[k])], starttime=UTCDateTime(ref_time) + begin_time,
                  endtime=UTCDateTime(ref_time) + end_time)
        kstnm_i = int(channel_sac_num[k])
        if kstnm_i - int(channel_sac_num[k-1]) != 1:
            fill_zeros = np.zeros(data.shape[0], dtype='float32')
            fill_zeros = fill_zeros
            data = np.vstack((data, fill_zeros))
        else:
            data = np.vstack((data, st[0].data))

    data = data.T

    return data, delta
