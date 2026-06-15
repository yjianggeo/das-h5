'''
Das processing for h5 data with command line.

By Hardy Hong 2023.07.16
'''

import os
import time
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
from obspy import read, Stream, Trace, UTCDateTime

from das_h5.io.h5_reader import make_dbfile, read_h5, read_sac
from das_h5.io.writer import save_sac, save_npz, save_sac_fast
from das_h5.io.sac_export import h5_to_sac
from das_h5.process.filters import filter, detrend, rmean, norm_data
from das_h5.process.resample import decimate, interpolate
from das_h5.viz.waterfall import das_imshow
from das_h5.viz.wiggle import das_fillplot
from das_h5.viz.trace import spectrograms, single_plot
from das_h5.psd.compute import das_psd, das_psd_type, das_psd_pkl
from das_h5.psd.plots import (
    plot_time_psd, plot_psd_freq, plot_psd_freq_time,
    plot_psd_channel_statistic, plot_psd_freq_time_single, plot_cure_single,
)
from das_h5.sound.audio import das_sound


def main():
    start_time = time.time()
    # the value to input
    parser = argparse.ArgumentParser(description="DAS plot for all data or single trace")
    parser.add_argument('-h5', '--h5_model', nargs='?', const=True, default=False, help='Input data model h5 file dir')
    parser.add_argument('-npy', '--npy_model', nargs='?', const=True, default=False, help='Input data model npy file')
    parser.add_argument('-sac', '--sac_model', nargs='?', const=True, default=False, help='Input data model sac file dir')
    parser.add_argument('-d', '--datadir', type=str, default=False, help='The data dir path or npy file, it must be needed')
    parser.add_argument('-idb', '--input_db', type=str, default=False, help='For h5_model, if db file have existed')
    parser.add_argument('-odb', '--out_db', type=str, default=False, help='For h5_model, if db file need to be made')
    parser.add_argument('-T', '--reftime', type=str, default=False, help='The data reference time')
    parser.add_argument('-b', '--starttime', type=float, default=0, help='The data relative start time from reference time')
    parser.add_argument('-e', '--endtime', type=float, default=0, help='The data relative end time from reference time')
    parser.add_argument('-c', '--startchannel', type=int, default=False, help='The data start channel that you need')
    parser.add_argument('-C', '--endchannel', type=int, default=False, help='The data end channel that you need')

    # save data values
    parser.add_argument('-so', '--sac_out', type=str, default=False, help='Out sac data dir path--optional')
    parser.add_argument('-sof', '--sac_out_fast', type=str, default=False, help='Out sac data dir path (faster) --optional')
    parser.add_argument('-ea', '--evla', type=float, default=False, help='Sac data evla (event latitude)--optional')
    parser.add_argument('-eo', '--evlo', type=float, default=False, help='Sac data evlo (event longitude)--optional')
    parser.add_argument('-ed', '--evdp', type=float, default=False, help='Sac data evdp (event depth)--optional')
    parser.add_argument('-loc', '--location', type=str, default=False, help='DAS channel locations--optional')
    parser.add_argument('-no', '--npy_out', type=str, default=False, help='Out npy data dir path--optional')
    parser.add_argument('-de', '--decimate', type=int, default=False, help='Decimate factor --optional')

    # general plot values
    parser.add_argument('-fz', '--fig_size', type=float, nargs=2, default=[10, 7], help='The all data fig size --optional')
    parser.add_argument('-sfz', '--single_size', type=float, nargs=2, default=[10, 6], help='The single fig size --optional')
    parser.add_argument('-dpi', '--fig_dpi', type=float, default=200, help='The fig dpi --optional')
    parser.add_argument('-v', '--vertical', nargs='?', const=True, default=False, help='The fig direction, if -v, vertical fig')
    parser.add_argument('-ti', '--figtitle', type=str, default=None, help='The part title of figs --optional')
    parser.add_argument('-ptm', '--plot_time', type=float, nargs=2, default=False, help='Plot time lim --optional')
    parser.add_argument('-pch', '--plot_channel', type=float, nargs=2, default=False, help='Plot channel lim --optional')

    # imshow plot
    parser.add_argument('-imo', '--imshow_out', type=str, default=False, help='Out imshow fig name (include path)--optional')
    parser.add_argument('-imc', '--imshow_cmap', type=str, default='RdBu', help='Imshow fig colormap--optional')
    parser.add_argument('-imn', '--imshow_norm', type=float, nargs=2, default=[-1, 1], help='Imshow fig colorbar norm--optional')
    parser.add_argument('-ims', '--imshow_show', nargs='?', const=True, default=False, help='The imshow fig shows--optional')
    parser.add_argument('-ng', '--norm_global', nargs='?', const=True, default=False, help='Use global (whole-array) normalization for section plots instead of per-channel--optional')

    # fill plot
    parser.add_argument('-flo', '--fill_out', type=str, default=False, help='Out fill color fig name (include path)--optional')
    parser.add_argument('-flc', '--fill_cmap', type=str, nargs=2, default=('r', 'b'), help='Fill color cmap--optional')
    parser.add_argument('-flg', '--fill_gain', type=float, default=2, help='Fill color wave scale--optional')
    parser.add_argument('-fls', '--fill_show', nargs='?', const=True, default=False, help='The fill color fig shows--optional')

    # single plot
    parser.add_argument('-num', '--channel_num', type=int, default=False, help='Which channel to plot--optional')
    parser.add_argument('-sio', '--single_out', type=str, default=False, help='Single plot figure to save--optional')
    parser.add_argument('-sro', '--single_range_out', type=str, default=False, help='Range single plot figures saving path--optional')
    parser.add_argument('-spe', '--spec_lim', type=float, nargs=2, default=False, help='Spec lim--optional')
    parser.add_argument('-spn', '--spec_norm', type=float, nargs=2, default=None, help='Spec norm--optional')
    parser.add_argument('-spes', '--single_show', nargs='?', const=True, default=False, help='The single trace fig shows--optional')
    parser.add_argument('-sfr', '--single_rate', type=float, default=0.25/0.8, help='Single show rate of ax--optional')

    # data processing
    parser.add_argument('-bp', '--bandpass', type=float, nargs=2, default=False, help='Bandpass filter--optional')
    parser.add_argument('-bs', '--bandstop', type=float, nargs=2, default=False, help='Bandstop filter--optional')
    parser.add_argument('-lp', '--lowpass', type=float, default=False, help='Lowpass filter--optional')
    parser.add_argument('-hp', '--highpass', type=float, default=False, help='Highpass filter--optional')
    parser.add_argument('-od', '--order', type=int, default=4, help='Filter order--optional')
    parser.add_argument('-zr', '--zero_phase', nargs='?', const=False, default=True, help='Need to apply forward and back--optional')

    # psd 2024.03.12  By Hardy Hong
    parser.add_argument('-wle', '--wind_len', type=float, default=False, help='Min freq band values--optional')
    parser.add_argument('-wst', '--wind_step', type=float, default=False, help='Max freq band values--optional')
    parser.add_argument('-mif', '--minfreq', type=float, nargs='+', default=False, help='Min freq band values--optional')
    parser.add_argument('-maf', '--maxfreq', type=float, nargs='+', default=False, help='Max freq band values--optional')
    parser.add_argument('-fnum', '--band_num', type=int, default=False, help='Which band to plot--optional')
    parser.add_argument('-chs', '--single_channel', type=int, default=False, help='Which single channel to plot, de to ci--optional')
    parser.add_argument('-chr', '--channel_range', type=int, nargs=2, default=False, help='Channel range to plot--optional')
    parser.add_argument('-chc', '--channel_choose', type=int, nargs='+', default=False, help='Choose channels to plot--optional')
    parser.add_argument('-f1', '--rms_time', type=str, default=False, help='All channels-times rms fig to save--optional')
    parser.add_argument('-f2', '--rms_freq', type=str, default=False, help='All channels-freq rms fig to save--optional')
    parser.add_argument('-f3', '--single_time', type=str, default=False, help='Channel freq-time fig to save--optional')
    parser.add_argument('-f4', '--more_freq', type=str, default=False, help='Channel more freq-time fig to save--optional')
    parser.add_argument('-f5', '--statistic_freq', type=str, default=False, help='Channel statistic freq fig to save--optional')
    parser.add_argument('-f6', '--more_channels', type=str, default=False, help='More channels fig to save--optional')
    parser.add_argument('-f1g', '--fig1_scale', type=float, default=6, help='Fig1 scale--optional')
    parser.add_argument('-f2g', '--fig2_scale', type=float, default=20, help='Fig2 scale--optional')
    parser.add_argument('-f3g', '--fig3_scale', type=float, default=2, help='Fig3 scale--optional')
    parser.add_argument('-bis', '--statistic_bins', type=int, default=100, help='Channel statistic bins to save --optional')
    parser.add_argument('-smh', '--fig_smooth', nargs='?', const=True, default=False, help='Figs smooth--optional')
    parser.add_argument('-lg', '--freq_log', nargs='?', const=True, default=False, help='Fig freq of log--optional')
    parser.add_argument('-pr', '--period_model', nargs='?', const=True, default=False, help='Fig freq to period--optional')
    parser.add_argument('-fs', '--figs_show', nargs='?', const=True, default=False, help='All channels rms fig to save--optional')
    parser.add_argument('-fall', '--figs_all', type=str, default=False, help='All figs to save--optional')
    parser.add_argument('-opd', '--out_psd', type=str, default=False, help='The path of output psd data--optional')
    parser.add_argument('-ipd', '--in_psd', type=str, default=False, help='The path of input psd data--optional')

    # DAS sound model
    parser.add_argument('-sds', '--sound_scale', type=float, default=1.0, help='Sound data sample scale-Sample*Scale--optional')
    parser.add_argument('-sdg', '--sound_gain', type=int, default=5000, help='Sound data scale-Data*Gain--optional')
    parser.add_argument('-sdo', '--sound_outpath', type=str, default=False, help='Out sound data path--optional')
    parser.add_argument('-sdi', '--sound_interp', type=int, default=False, help='Whether to interp data, the interp scale--optional')


    args = parser.parse_args()

    def usage():
        print('USEAGE : python3 DASh5.py -h5/-sac/-npy -d data_dir -T 20230520T131420.5 (20230520131420.5) -b 0 -e 10'
              '-c 0 -C 10 -im imshow_fig_name -fl fill_fig_name -spe single_fig_name -nu 2 \
              (if it use) -so sac_path -no npy_path -bp [0, sample/2] | -lp [1] | -hp [1] -flg [4] -v and so on ...')

    if not(args.h5_model or args.sac_model or args.npy_model) :
        usage()

    if not(args.datadir or args.reftime):
        usage()

    # global values
    data_dir = args.datadir
    if args.reftime:
        ref_time = UTCDateTime(args.reftime)
        # print(ref_time)
    else:
        ref_time = args.reftime
    b = args.starttime
    e = args.endtime
    if args.channel_num and not(args.startchannel or args.endchannel):
        ci = args.channel_num
        ca = args.channel_num
    else:
        ci = args.startchannel
        ca = args.endchannel
    dpi = args.fig_dpi
    fig_size = [args.fig_size[0], args.fig_size[1]]
    single_size = [args.single_size[0], args.single_size[1]]
    if args.spec_norm:
        single_norm = [args.spec_norm[0], args.spec_norm[1]]
    else:
        single_norm = None
    if args.vertical:
        direction = 'vertical'
    else:
        direction = 'horizontal'
    if args.plot_time:
        plot_time = [args.plot_time[0], args.plot_time[1]]
    else:
        plot_time = [b, e]
    if args.plot_channel:
        plot_channel = [args.plot_channel[0], args.plot_channel[1]]
    else:
        plot_channel = [ci, ca]

    # for h5 data model
    if args.h5_model:
        print('\nRunning DAS h5 model ...')
        if args.input_db:
            if not os.path.exists(args.input_db):
                print('\nThe db file was wrong!!!')
                print('>>>···<<<')
                print(args.input_db + ' does not exist!!!')
                print('\nAlready Stopped!!!')
                return
            delta = None
            if ref_time:
                try:
                    if args.sac_out_fast:  # By Hardy Hong 2024.04.19
                        # if only out put sac data (large amount)
                        print('\nOnly convert h5 data to sac from the existed database file ...')
                        delta = h5_to_sac(db_file=args.input_db, time_flag=ref_time, begin_time=b, end_time=e,
                                    channel_begin=ci, channel_end=ca, location_data=args.location,
                                    event_la=args.evla, event_lo=args.evlo,  event_depth=args.evdp,
                                    sac_path=args.sac_out_fast, deci=args.decimate)
                    else:
                        print('\nGetting h5 data from the existed database file ...')
                        data, delta = read_h5(db_file=args.input_db, time_flag=ref_time, begin_time=b, end_time=e,
                                       channel_begin=ci, channel_end=ca)
                except Exception as re:
                    print('\nGetting h5 data from the existed database file ... wrong!')

        else:
            if args.out_db:
                db_file = args.out_db
                print('\nMaking database file and saving it ...')
            else:
                db_file = None
                print('\nMaking database file but do not saving it ...')
            delta = None

            if not(os.path.isdir(data_dir)):
                print('\nThe data dir was wrong!!!')
                print('>>>···<<<')
                print(data_dir + ' does not exist!!!')
                print('\nAlready Stopped!!!')
                return
            try:
                db_list = make_dbfile(data_dir, outfile=db_file)
            except Exception as re:
                print('\nMaking database ... wrong!')
            if ref_time:
                try:
                    if args.sac_out_fast:  # By Hardy Hong 2024.04.19
                        # if only out put sac data (large amount)
                        print('\nOnly convert h5 data to sac from the database file made right now ...')
                        delta = h5_to_sac(db_list=db_list, time_flag=ref_time, begin_time=b, end_time=e,
                                  channel_begin=ci, channel_end=ca, location_data=args.location,
                                  event_la=args.evla, event_lo=args.evlo, event_depth=args.evdp,
                                  sac_path=args.sac_out_fast, deci=args.decimate)
                    else:
                        print('\nGetting h5 data from the database file made right now ...')
                        data, delta = read_h5(db_list=db_list, time_flag=ref_time, begin_time=b, end_time=e,
                                       channel_begin=ci, channel_end=ca)
                except Exception as re:
                    print('\nGetting h5 data from the database file made right now. ... wrong!')


    # get data from npy model
    if args.npy_model:
        print('\nRunning DAS npy model ...')
        data = np.load(data_dir)
        name = os.path.split(args.datadir)[1]
        npy_reftime = UTCDateTime(name.split('.')[2])
        if name.split('.')[1][0:2] == 'SP':
            npy_b = float(name.split('.')[1][2:5])
        else:
            npy_b = -float(name.split('.')[1][2:5])
        delta = 1 / float(name.split('.')[3])
        npy_ci = int(name.split('.')[4])
        npy_ca = int(name.split('.')[5])

        if ref_time + b >= npy_reftime + npy_b and ref_time + e - 3 * delta <= npy_reftime + npy_b + \
                delta * data.shape[0]:
            time_range = True
        else:
            print('\nTime range is out raw time range!')
            return

        if npy_ci <= ci and npy_ca >= ca:
            channel_range = True
        else:
            print('\nChannel range  is out raw channel range!')
            return

        start_point = int((ref_time + b - (npy_reftime + npy_b)) / delta)
        end_point = int(start_point + (e - b) / delta)
        data = data[start_point:end_point, ci:ca+1]

    # sac data model
    if args.sac_model:
        print('\nRunning DAS sac model ...')
        sac_list = os.listdir(data_dir)
        sac_list.sort()
        st = Stream()
        st_example = read(data_dir + sac_list[0])
        delta = st_example[0].stats.delta

        if ref_time + b >= st_example[0].stats.starttime and ref_time + e - 3 * delta <= st_example[0].stats.endtime:
            time_range = True
        else:
            print('\nTime range is out raw time range!')
            return

        if int(sac_list[0].split('.')[1]) <= ci and int(sac_list[-1].split('.')[1]) >= ca:
            channel_range = True
        else:
            print('\nChannel range is out raw channel range!')
            return

        # get all or part sac data
        if args.imshow_out or args.imshow_show or args.fill_out or args.fill_show:
            print('\nGetting all data from sac file ...')
            data, delta = read_sac(data_dir, ref_time=ref_time, begin_time=b, end_time=e,
                                   begin_channel=ci, end_channel=ca)


        # get single trace and plot
        if args.channel_num or ((ca - ci) == 0 and ca):   # 01.22
            if not(args.imshow_out or args.imshow_show or args.fill_out or args.fill_show):
                print('\nOnly plotting the single trace fig ...')
                if ca - ci == 0:
                    channel_name = ca
                else:
                    channel_name = args.channel_num
                n = str(channel_name)
                sac_single = read(data_dir + '*.' + n.zfill(5) + '.*.sac')
                sac_single.trim(starttime=ref_time + b, endtime=ref_time + e)
                data = sac_single[0].data     # be easy to use filter
                data = data.reshape(-1, 1)
                delta = sac_single[0].stats.delta
                end_example = sac_single[0].stats.endtime

    # h5 model or npy model process
    if args.h5_model or args.npy_model or args.sac_model:
        if ref_time:
            ms = ref_time.strftime('%f')
            if args.figtitle:
                title = args.figtitle + ' DAS records | ' + ref_time.strftime("%Y-%m-%dT%H:%M:%S.") + ms[0:2]
            else:
                title = 'DAS records | ' + ref_time.strftime("%Y-%m-%dT%H:%M:%S.") + ms[0:2]

        if args.decimate and not args.sac_out_fast:
            print('\nData decimating ...')
            de_factor = args.decimate
            data = decimate(data, delta=delta, factor=de_factor, axis=0)
            delta = delta * float(de_factor)

        if delta is not None:
        	spec_lim = [0, 1 / delta * 0.5]  # default spec_lim

        if args.bandpass:
            print('\nRuning bandpass filter ...')
            data = detrend(data)
            data = filter(data, delta=delta, type='bandpass', freq=(args.bandpass[0], args.bandpass[1]),
                          order=args.order, zerophase=args.zero_phase)
            if args.bandpass[0] % 1 == 0:
                hz1 = str(int(args.bandpass[0]))
            else:
                hz1 = str(args.bandpass[0])
            if args.bandpass[1] % 1 == 0:
                hz2 = str(int(args.bandpass[1]))
            else:
                hz2 = str(args.bandpass[1])
            title = title + ' | BP ' + hz1 + '-' + hz2 + \
                    ' Hz' + ' | Order ' + str(int(args.order))
            spec_lim = [args.bandpass[0], args.bandpass[1]]

        elif args.bandstop:
            data = detrend(data)
            print('\nRuning bandstop filter ...')

            data = filter(data, delta=delta, type='bandpass', freq=(args.bandpass[0], args.bandpass[1]),
                          order=args.order, zerophase=args.zero_phase)
            title = title + ' | BS ' + str(int(args.bandpass[0])) + '-' + str(int(args.bandpass[1])) + \
                    ' Hz' + ' | Order ' + str(int(args.order))
        elif args.lowpass:
            data = detrend(data)
            print('\nRuning lowpass filter ...')
            if args.lowpass % 1 == 0:
                hz1 = str(int(args.lowpass))
            else:
                hz1 = str(args.lowpass)
            data = filter(data, delta=delta, type='lowpass', freq=args.lowpass,
                          order=args.order, zerophase=args.zero_phase)
            title = title + ' | LP ' + hz1 + ' Hz' + ' | Order ' + str(int(args.order))
            spec_lim = [0, args.lowpass]

        elif args.highpass:
            data = detrend(data)
            print('\nRuning highpass filter ...')
            if args.highpass % 1 == 0:
                hz1 = str(int(args.highpass))
            else:
                hz1 = str(args.highpass)
            data = filter(data, delta=delta, type='highpass', freq=args.highpass,
                          order=args.order, zerophase=args.zero_phase)
            title = title + ' | HP ' + hz1 + ' Hz' + ' | Order ' + str(int(args.order))
            spec_lim = [args.highpass, 1 / delta * 0.5]

        if args.sac_out:
            print('\nSaving Sac data ...')
            save_sac(data, time_flag=ref_time, delta=delta, begin_time=b, channel_begin=ci, channel_end=ca,
                     location_data=args.location, event_la=args.evla, event_lo=args.evlo, event_depth=args.evdp,
                     outpath=args.sac_out)

        if args.npy_out:
            print('\nSaving Numpy array data ...')
            if b >= 0:
                bs = str('SP')
            else:
                bs = str('SN')
            bt = str(int(b)).zfill(3)
            if not(os.path.exists(args.npy_out)):
                os.makedirs(args.npy_out)
            outnpy = args.npy_out + 'DAS.' + bs + bt + '.' + ref_time.strftime('%Y%m%d%H%M%S') + '.' + \
                        str(int(1/delta)).zfill(5) + '.' + str(ci).zfill(5) + '.' + str(ca).zfill(5) + '.npy'
            save_npz(data, outnpy=outnpy)

        if args.channel_num or ((ca - ci) == 0 and ca): # 2023.01.22
            if (ca - ci) == 0:
                channel_name = ca
            else:
                channel_name = args.channel_num
            print('\nPlotting single Trace-'+str(channel_name)+' ...')
            titles = title + ' | Trace-' + str(channel_name)
            data_s = data[:, channel_name - ci]
            if args.spec_lim:
                spec_lim = [args.spec_lim[0], args.spec_lim[1]]
            else:
                spec_lim = spec_lim

            single_plot(data_s, delta=delta, begin_time=b, end_time=e, title=titles, time_lim=plot_time,
                        spec_lim=spec_lim, outfig=args.single_out, show=args.single_show, dpi=dpi, figsize=single_size,
                        norm=single_norm, show_rate=args.single_rate)

            # DAS sound model 2024.03.29
            if args.sound_outpath:
                print('\nWriting Sound(.wav) in sample-' + str(int(1/delta*args.sound_scale)) + ' ...')
                sound_time = ref_time + b
                title_sound = 'Sound.' + str(channel_name).zfill(5) + '.' + sound_time.strftime('%Y%m%d%H%M%S')
                das_sound(data=data_s, delta=delta, scale=args.sound_scale, save_path=args.sound_outpath,
                          interp=args.sound_interp, title=title_sound, gain=args.sound_gain)
                print('Writing ' + args.sound_outpath + title_sound + '.wav Successfully!')

        # plot more single figs at one time
        if args.single_range_out:
            if not(os.path.exists(args.single_range_out)):
                os.makedirs(args.single_range_out)
            print('\nPlotting single Trace in range ' + str(ci) + '-' + str(ca) + ' ...')
            for i in range(ci, ca+1):
                print('Plotting single Trace-' + str(i) + ' ...')
                titles = title + ' | Trace-' + str(i)
                data_s = data[:, i - ci]
                if args.spec_lim:
                    spec_lim = [args.spec_lim[0], args.spec_lim[1]]
                else:
                    spec_lim = spec_lim
                fig_out = args.single_range_out + 'Trace-' + str(i) + '.png'
                single_plot(data_s, delta=delta, begin_time=b, end_time=e, title=titles, time_lim=plot_time,
                            spec_lim=spec_lim, outfig=fig_out, show=False, dpi=dpi, figsize=single_size,
                            norm=single_norm, show_rate=args.single_rate)

        if args.imshow_show or args.imshow_out or args.fill_show or args.fill_out:
            norm_type = 'all' if args.norm_global else 'self'
            data = norm_data(data, type=norm_type)

        if args.imshow_show or args.imshow_out:
            print('\nPlotting imshow figure ...')
            das_imshow(data, delta=delta, begin_time=b, end_time=e, data_norm=args.imshow_norm, time_lim=plot_time,
                       channel_begin=ci, channel_end=ca, title=title, colormap=args.imshow_cmap, channel_lim=plot_channel,
                       dpii=dpi, figsize_arr=fig_size, direction=direction, outfig=args.imshow_out, show=args.imshow_show)

        if args.fill_show or args.fill_out:
            print('\nPlotting fill figure ...')
            das_fillplot(data, delta=delta, begin_time=b, end_time=e, channel_begin=ci, channel_end=ca, title=title, dpii=dpi,
                         fillcolor=(str(args.fill_cmap[0]), str(args.fill_cmap[1])), gain=args.fill_gain, direction=direction,
                         figsize_arr=fig_size, time_lim=plot_time, channel_lim=plot_channel, outfig=args.fill_out,
                         show=args.fill_show)


    # PSD Model. 2024.03.13 By Hardy Hong
    if args.wind_len and args.wind_step and args.minfreq and args.maxfreq:
        print('\nRunning PSD Model ...')
        utt = ref_time + b
        minf = args.minfreq
        maxf = args.maxfreq
        if args.figs_show:
            show = True

        if args.single_channel:
            chss = args.single_channel
        else:
            chss = ci

        # Whether to output psd
        if args.in_psd:
            print('\nReading PSD Values ...')
            fss, psds, times, _ = das_psd_pkl(pkl_path=args.in_psd, ch1=ci, ch2=ca)
        else:
            if args.out_psd:
                print('\nCalculating PSD Values ...')
                fss, psds, times, _ = das_psd(das_data=data, delta=delta, wind_len=args.wind_len, wind_step=args.wind_step,
                                              minfreq=args.minfreq, maxfreq=args.maxfreq, scaling='density',
                                              starttime=UTCDateTime(utt), ch1=ci, out_psd=args.out_psd)
            else:
                print('\nCalculating PSD Values ...')
                fss, psds, times, _ = das_psd(das_data=data, delta=delta, wind_len=args.wind_len, wind_step=args.wind_step,
                                              minfreq=args.minfreq, maxfreq=args.maxfreq, scaling='density',
                                              starttime=UTCDateTime(utt), ch1=ci)

        if args.minfreq:
            if type(args.minfreq) is float:
                num = 0
                freq_band = 1
                minf = [args.minfreq]
                maxf = [args.maxfreq]
            else:
                minf = args.minfreq
                maxf = args.maxfreq
                freq_band = len(minf)
                num = args.band_num - 1
        else:
            num = 0
            freq_band = 1
            minf = [0.01]
            maxf = [1/delta/2]


        psds_rms, psds_freq_all, psds_singel_time, psds_singel_time_freq_band, \
        psds_single_band, psds_single_time_psds, psds_single_time_freqs = das_psd_type(fss, psds, times,
                                                                                       freq_band, num, ci,
                                                                                chss, args.statistic_bins)

        # print('\nPlotting PSD Figs ...')
        # Plot PSD
        # Fig1
        if args.rms_time or args.figs_show:
            print('\nPlotting PSD Fig Type-1 ...')
            plot_time_psd(times=times, psds_rms=psds_rms.copy(), maxfreq=maxf[num], minfreq=minf[num], min_channel=ci,
                          max_channel=ca, figsave=args.rms_time, scale=args.fig1_scale, smooth=args.fig_smooth,
                          show=args.figs_show, dpi=dpi)


        # Fig2
        if args.rms_freq or args.figs_show:
            print('\nPlotting PSD Fig Type-2 ...')
            plot_psd_freq(times=times, psds_freq=psds_freq_all / len(times), minfreq=minf[num], maxfreq=maxf[num],
                          min_channel=ci,max_channel=ca, figsave=args.rms_freq, scale=args.fig2_scale,
                          smooth=args.fig_smooth, show=args.figs_show, dpi=dpi)


        # Fig3
        if args.single_time or args.figs_show:
            print('\nPlotting PSD Fig Type-3 ...')
            plot_psd_freq_time(times=times, psds_single=psds_singel_time, maxfreq=maxf[num], minfreq=minf[num],
                               channel=chss, figsave=args.single_time, scale=args.fig3_scale, smooth=args.fig_smooth,
                               show=args.figs_show, dpi=dpi)


        # Fig4
        if args.more_freq or args.figs_show:
            print('\nPlotting PSD Fig Type-4 ...')
            plot_psd_freq_time_single(times=times, psds_single=psds_single_band,
                                      minfreq=minf, maxfreq=maxf, show=args.figs_show,
                                      channel=chss, figsave=args.more_freq, dpi=dpi)


        # Fig5
        if args.statistic_freq or args.figs_show:
            print('\nPlotting PSD Fig Type-5 ...')
            plot_psd_channel_statistic(times=times, psds_channel=psds_single_time_psds, fss_channel=psds_single_time_freqs,
                                       channel=chss, bins=args.statistic_bins, fss_count=len(fss[0][num]),
                                       smooth=False, show=args.figs_show, figsave=args.statistic_freq,
                                       period=False, segment=args.wind_len, types='pqlx', freq_log=args.freq_log, dpi=dpi)


        # Fig6
        if args.channel_range:
            ch_choose_range = np.arange(args.channel_range[0], args.channel_range[1]+1, 1)
        elif args.channel_choose:
            ch_choose_range = np.array(args.channel_choose)
        else:
            ch_choose_range = [ci, ci+1, ci+2]

        if args.more_channels:
            print('\nPlotting PSD Fig Type-6 ...')
            plot_cure_single(times=times, psds_single=10 * np.log10(psds_freq_all[:, ch_choose_range - ci] / len(times)),
                             channel=ch_choose_range, fss=fss[0][num], freq_scale=args.freq_log, period=args.period_model,
                             figsave=args.more_channels, figsize=(7.8, 6), show=args.figs_show, dpi=dpi)

        # Fig all - only save figs
        if args.figs_all:
            print('\nPlotting All PSD Figs ...')
            if not os.path.exists(args.figs_all):
                os.mkdir(args.figs_all)
            title_psd = utt.strftime('%Y%m%d%H%M%S-')
            plot_time_psd(times=times, psds_rms=psds_rms.copy(), maxfreq=maxf[num], minfreq=minf[num], min_channel=ci,
                          max_channel=ca, figsave=args.figs_all+'psd-'+title_psd+'fig1.png', scale=args.fig1_scale,
                          smooth=args.fig_smooth, show=args.figs_show, dpi=dpi)
            plot_psd_freq(times=times, psds_freq=psds_freq_all / len(times), minfreq=minf[num], maxfreq=maxf[num],
                          min_channel=ci, max_channel=ca, figsave=args.figs_all+'psd-'+title_psd+'fig2.png',
                          scale=args.fig2_scale, smooth=args.fig_smooth, show=args.figs_show, dpi=dpi)
            plot_psd_freq_time(times=times, psds_single=psds_singel_time, maxfreq=maxf[num], minfreq=minf[num],
                               channel=chss, figsave=args.figs_all+'psd-'+title_psd+'fig3.png', scale=args.fig3_scale,
                               smooth=args.fig_smooth, show=args.figs_show, dpi=dpi)
            plot_psd_freq_time_single(times=times, psds_single=psds_single_band,
                                      minfreq=minf, maxfreq=maxf, show=args.figs_show,
                                      channel=chss, figsave=args.figs_all+'psd-'+title_psd+'fig4.png', dpi=dpi)
            plot_psd_channel_statistic(times=times, psds_channel=psds_single_time_psds, fss_channel=psds_single_time_freqs,
                                       channel=chss, bins=args.statistic_bins, fss_count=len(fss[0][num]),
                                       smooth=False, show=args.figs_show, figsave=args.figs_all+'psd-'+title_psd+'fig5.png',
                                       period=False, segment=args.wind_len, types='pqlx', freq_log=args.freq_log, dpi=dpi)
            plot_cure_single(times=times, psds_single=10 * np.log10(psds_freq_all[:, ch_choose_range - ci] / len(times)),
                             channel=ch_choose_range, fss=fss[0][num], freq_scale=args.freq_log, period=args.period_model,
                             figsave=args.figs_all+'psd-'+title_psd+'fig6.png', figsize=(7.8, 6), show=args.figs_show, dpi=dpi)


    end_time = time.time()
    print('\nDone! Time is ', str('%.2f'%(end_time-start_time)), 's!')
    plt.clf()
    return


if __name__ == "__main__":
    main()
