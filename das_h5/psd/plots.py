"""Plot DAS PSD results: time/freq/channel figures (six figure types).

Repackaged from DASh5 das_psd_plot.py (By Hardy Hong). Plotting unchanged.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker

def plot_time_psd(times, psds_rms, min_channel, max_channel, minfreq, maxfreq, smooth=False,
        title=None, figsize=None, dpi=300, figsave=None, show=False, cmap=None, norm=None,
        scale=4, time_ticks=10, vertical=False):
    '''
    Plot time psd of das channels.
    By Hardy Hong 2024.01.15

    :param times:           time series [list] (UTCDateTime)
    :param psds_rms:        psds rms [list] with freq band
    :param min_channel:     min channel
    :param max_channel:     max channel
    :param minfreq:         min frequency
    :param maxfreq:         max frequency
    :param smooth:          whether to smooth
    :param title:           head of title
    :param figsize:         figsize
    :param dpi:             dpi
    :param figsave:         save fig file
    :param show:            whether to show
    :param cmap:            colorbar
    :param norm:            [vmin, vmax]
    :param scale:           [data_min, data_max/scale], bigger scale, show more things
    :param time_ticks:      if show datetime ticks, choosing amount ticks to show
    :param vertical:        if true, plot fig vertically.
    :return:                None
    '''

    if figsize is None:
        if times[-1] - times[0] > 300:
            figsize = [10, 6.5]
        else:
            figsize = [10, 5.5]
    if not cmap:
        cmap = 'jet'
    time_plot_xticks = []
    nums = int(len(times) / time_ticks)
    choose_num = np.arange(20) * nums
    k = 0
    ylim = [min_channel, max_channel]
    if times[-1] - times[0] > 300:
        for i in choose_num:
            if i < len(times):
                if i == 1:
                    str_time_xtick = times[i].strftime("%Y-%m-%dT%H:%M:%S")
                else:
                    str_time_xtick = times[i].strftime("%d..%H:%M:%S")
                time_plot_xticks.append(str_time_xtick)
                k += 1
            else:
                break
        choose_num = choose_num[0:k]
        if vertical:
            extent = [min_channel, max_channel, len(times), 0]
            ylim = [len(times), 0]
        else:
            extent = [0, len(times), max_channel, min_channel]
    else:
        dt = times[1] - times[0]
        times_s = np.arange(len(times)) * dt + dt/2
        if vertical:
            extent = [min_channel, max_channel, times_s[-1], times_s[0]]
            ylim = [times_s[-1], times_s[0]]
        else:
            extent = [times_s[0], times_s[-1], max_channel, min_channel]


    psds_rms = psds_rms.T
    if not norm:
        norm = [np.min(np.min(psds_rms)), np.max(np.max(psds_rms))/scale]

    # plot psd
    fig = plt.figure(figsize=figsize, dpi=dpi)
    if times[-1] - times[0] > 300:
        ax = plt.axes([0.104, 0.15, 0.805, 0.77])
        cax = fig.add_axes([0.918, 0.15, 0.01, ax.get_position().height])
    else:
        ax = plt.axes([0.104, 0.12, 0.805, 0.80])
        cax = fig.add_axes([0.918, 0.12, 0.01, ax.get_position().height])
    if smooth:
        if vertical:
            psd_das = ax.imshow(psds_rms.T, cmap=cmap, extent=extent,
                                vmin=norm[0], vmax=norm[1], aspect='auto',
                                interpolation='hanning')
        else:
            psd_das = ax.imshow(psds_rms, cmap=cmap, extent=extent,
                                vmin=norm[0], vmax=norm[1], aspect='auto',
                                interpolation='hanning')
    else:
        if vertical:
            psd_das = ax.imshow(psds_rms.T, cmap=cmap, extent=extent,
                                vmin=norm[0], vmax=norm[1], aspect='auto')
        else:
            psd_das = ax.imshow(psds_rms, cmap=cmap, extent=extent,
                                vmin=norm[0], vmax=norm[1], aspect='auto')

    if title:
        title = title + ' RMS-PSD' + ' | ' + times[0].strftime("%Y-%m-%dT%H:%M:%S") + ' | ' \
                + str(minfreq) + '-' + str(maxfreq) + ' Hz'
    else:
        title = 'RMS-PSD' + ' | ' + times[0].strftime("%Y-%m-%dT%H:%M:%S") + ' | ' \
                + str(minfreq) + '-' + str(maxfreq) + ' Hz'
    ax.set_title(title, fontsize=18)
    if vertical:
        ax.set_xlabel('Channel [number]', fontsize=16)
    else:
        ax.set_ylabel('Channel [number]', fontsize=16)
    ax.set_ylim(ylim)
    plt.setp(ax.get_xticklabels(), size=15)
    plt.setp(ax.get_yticklabels(), size=15)

    # for colorbar
    cb = plt.colorbar(psd_das, ax=ax, cax=cax)
    cb_label = 'RMS Amp'
    cb.ax.set_ylabel(cb_label, fontsize=13)
    tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar ticks number
    cb.locator = tick_locator
    cb.ax.tick_params(labelsize=12.5)
    cb.ax.ticklabel_format(style='sci', scilimits=(-1, 1))
    # cb.ax.set_title('unit', fontsize=12.5)
    cb.update_ticks()

    if times[-1] - times[0] > 300:
        # ax.ylabel('Time')
        if vertical:
            ax.set_yticks(choose_num[1:])
            ax.set_yticklabels(time_plot_xticks[1:],
                               horizontalalignment='right', rotation=30, fontsize=13)
            ax.invert_yaxis()
        else:
            ax.set_xticks(choose_num[1:])
            ax.set_xticklabels(time_plot_xticks[1:],
                               horizontalalignment='right', rotation=30, fontsize=13)
    else:
        if vertical:
            ax.set_ylabel('Time [s]', fontsize=16)
            ax.invert_yaxis()
        else:
            ax.set_xlabel('Time [s]', fontsize=16)
    if show:
        plt.show()
    elif figsave:
        plt.savefig(figsave)
    else:
        plt.clf()
        plt.close()

def plot_psd_freq(psds_freq, times, minfreq, maxfreq, min_channel, max_channel,
                  title=None, cmap=None, figsize=(8, 5), dpi=300, figsave=None,
                  norm=None, show=False, smooth=False, scale=9, vertical=False):
    '''
    Plot psd of freq of channels.
    By Hardy Hong 204.01.16

    :param psds_freq:       psds in freq band.    [freq, channel] array
    :param times:           times [list] UTCDateTime
    :param minfreq:         min frequency
    :param maxfreq:         max frequency
    :param min_channel:     min channel
    :param max_channel:     max channel
    :param title:           head of title
    :param cmap:            colorbar, default to 'jet'
    :param figsize:         figsize
    :param dpi:             dpi
    :param figsave:         save fig file
    :param norm:            [vmin, vmax]
    :param show:            whether to show
    :param smooth:          wether to smooth fig
    :param scale:           [data_min, data_max/scale], bigger scale, show more things
    :param vertical:        if true, plot fig vertically.
    :return:                None
    '''

    if not cmap:
        cmap = 'jet'
    if vertical:
        extent = [minfreq, maxfreq, max_channel, min_channel]
    else:
        extent = [min_channel, max_channel, maxfreq, minfreq]


    if not norm:
        norm = [np.min(np.min(psds_freq)), np.max(np.max(psds_freq))/scale]

    # plot psd
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes([0.11, 0.12, 0.775, 0.80])
    cax = fig.add_axes([0.91, 0.12, 0.01, ax.get_position().height])
    if smooth:
        smooth = 'hanning'
    else:
        smooth = 'none'
    if vertical:
        psd_das = ax.imshow(psds_freq.T, cmap=cmap, extent=extent,
                            vmin=norm[0], vmax=norm[1], aspect='auto',
                            interpolation=smooth)
    else:
        psd_das = ax.imshow(psds_freq, cmap=cmap, extent=extent,
                            vmin=norm[0], vmax=norm[1], aspect='auto',
                            interpolation=smooth)

    if title:
        title = title + ' PSD' + ' | ' + times[0].strftime("%Y-%m-%dT%H:%M:%S") + '......' + \
                times[-1].strftime("%Y-%m-%dT%H:%M:%S")
    else:
        title = 'PSD' + ' | ' + times[0].strftime("%Y-%m-%dT%H:%M:%S") + '......' +\
                times[-1].strftime("%Y-%m-%dT%H:%M:%S")
    ax.set_title(title, fontsize=15)
    if vertical:
        ax.set_xlabel('Frequency [Hz]', fontsize=15)
        ax.set_ylabel('Channel [number]', fontsize=15)
        ax.set_ylim([max_channel, min_channel])
    else:
        ax.set_ylabel('Frequency [Hz]', fontsize=15)
        ax.set_xlabel('Channel [number]', fontsize=15)
        ax.set_ylim([minfreq,  maxfreq])
    plt.setp(ax.get_xticklabels(), size=14)
    plt.setp(ax.get_yticklabels(), size=14)

    # for colorbar
    cb = plt.colorbar(psd_das, ax=ax, cax=cax)
    cb_label = 'Amp'
    cb.ax.set_ylabel(cb_label, fontsize=12)
    tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar ticks number
    cb.locator = tick_locator
    cb.ax.tick_params(labelsize=11.5)
    cb.ax.ticklabel_format(style='sci', scilimits=(-1, 1))
    # cb.ax.set_title('unit', fontsize=12.5)
    cb.update_ticks()

    if show:
        plt.show()
    elif figsave:
        plt.savefig(figsave)
    else:
        plt.clf()
        plt.close()

def plot_cure_single(psds_single, fss, times, channel, title=None, figsize=(8, 6), dpi=300,
                     figsave=None, show=False, period=False, freq_scale=False):
    '''
    Plot PSD [dB] in freqs of channel or channels.
    By Hardy Hong 204.01.16

    :param psds_single:     PSDs [list with channels] in freqs
    :param fss:             freq series (array)
    :param times:           times [list UTCDateTime]
    :param channel:         channel number [list] or (array)
    :param title:           Ditto
    :param figsize:         Ditto
    :param dpi:             Ditto
    :param figsave:         Ditto
    :param show:            Ditto
    :param period:          If true plot period.
    :param freq_scale:      If true plot freq in log10.
    :return:                None
    '''

    # plot psd
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes([0.145, 0.12, 0.795, 0.80])
    if period:
        for i in range(len(channel)):
            ax.plot(1/fss, psds_single[:, i], '-', linewidth=1.5, label=str(channel[i]))
    else:
        for i in range(len(channel)):
            ax.plot(fss, psds_single[:, i], '-', linewidth=1.5, label=str(channel[i]))

    if title:
        title = title + ' PSD' + ' | ' + times[0].strftime("%Y-%m-%dT%H:%M:%S") + '...' + \
                times[-1].strftime("%Y-%m-%dT%H:%M:%S") + ' | ' + 'Channel'
    else:
        title = 'PSD' + ' | ' + times[0].strftime("%Y-%m-%dT%H:%M:%S") + '...' + \
                times[-1].strftime("%Y-%m-%dT%H:%M:%S") + ' | ' + 'Channel'
    ax.set_title(title, fontsize=15)

    ax.set_ylabel('PSD [$StrainRate^2/Hz$] [dB]', fontsize=15)
    ax.legend()
    plt.setp(ax.get_xticklabels(), size=14)
    plt.setp(ax.get_yticklabels(), size=14)
    if period:
        ax.set_xlabel('Period [s]', fontsize=15)
        ax.set_xscale('log')
    else:
        ax.set_xlabel('Frequency [Hz]', fontsize=15)
        if freq_scale:
            ax.set_xscale('log')
    if show:
        plt.show()
    elif figsave:
        plt.savefig(figsave)
        plt.clf()
        plt.close()

def plot_psd_freq_time(psds_single, times, minfreq, maxfreq, channel, cmap=None, norm=None, smooth=False,
                       figsave=None, show=False, timeticks=10, figsize=None, title=None, dpi=300, scale=9,
                       vertical=False):
    '''
    Plot single channel psd of times in freq.
    By Hardy Hong 2024.01.17.
    F-T fig.

    :param psds_single:     single channel PSD with time and freq.  [freq_psd_array1,...,...] is list.
    :param times:           times [list] UTCDateTime
    :param minfreq:         Ditto
    :param maxfreq:         Ditto
    :param channel:         Ditto
    :param cmap:            Ditto
    :param norm:            Ditto
    :param smooth:          Ditto
    :param figsave:         Ditto
    :param show:            Ditto
    :param timeticks:       Ditto
    :param figsize:         Ditto
    :param title:           Ditto
    :param dpi:             Ditto
    :param scale:           Ditto
    :param vertical         Ditto
    :return:                None
    '''

    if not cmap:
        cmap = 'jet'
    if figsize is None:
        if times[-1] - times[0] > 300:
            figsize = [10, 6.5]
        else:
            figsize = [10, 5.5]

    time_plot_xticks = []
    nums = int(len(times) / timeticks)
    choose_num = np.arange(20) * nums
    k = 0
    if times[-1] - times[0] > 300:
        for i in choose_num:
            if i < len(times):
                if i == 1:
                    str_time_xtick = times[i].strftime("%Y-%m-%dT%H:%M:%S")
                else:
                    str_time_xtick = times[i].strftime("%d..%H:%M:%S")
                time_plot_xticks.append(str_time_xtick)
                k += 1
            else:
                break
        choose_num = choose_num[0:k]
        if vertical:
            extent = [minfreq, maxfreq, len(times), 0]
        else:
            extent = [0, len(times), maxfreq, minfreq]
    else:
        dt = times[1] - times[0]
        times_s = np.arange(len(times)) * dt + dt/2
        if vertical:
            extent = [minfreq, maxfreq, times_s[-1], times_s[0]]
        else:
            extent = [times_s[0], times_s[-1], maxfreq, minfreq]


    if not norm:
        norm = [np.min(np.min(psds_single)), np.max(np.max(psds_single))/scale]

    # plot psd
    fig = plt.figure(figsize=figsize, dpi=dpi)
    if times[-1] - times[0] > 300:
        ax = plt.axes([0.095, 0.15, 0.80, 0.77])
        cax = fig.add_axes([0.91, 0.15, 0.01, ax.get_position().height])
    else:
        ax = plt.axes([0.095, 0.12, 0.80, 0.80])
        cax = fig.add_axes([0.91, 0.12, 0.01, ax.get_position().height])

    if smooth:
        smooth = 'hanning'
    else:
        smooth = 'none'

    if vertical:
        psd_das = ax.imshow(psds_single.T, cmap=cmap, extent=extent,
                            vmin=norm[0], vmax=norm[1], aspect='auto',
                            interpolation=smooth)
    else:
        psd_das = ax.imshow(psds_single, cmap=cmap, extent=extent,
                            vmin=norm[0], vmax=norm[1], aspect='auto', interpolation=smooth)

    if title:
        title = title + ' PSD' + ' | ' + times[0].strftime("%Y-%m-%dT%H:%M:%S") + ' | ' \
                + 'Channel-' + str(channel)
    else:
        title = 'PSD' + ' | ' + times[0].strftime("%Y-%m-%dT%H:%M:%S") + ' | ' \
                + 'Channel-' + str(channel)
    ax.set_title(title, fontsize=18)
    if vertical:
        ax.set_xlabel('Frequency [Hz]', fontsize=16)
        ax.set_ylim([minfreq, maxfreq])
        plt.setp(ax.get_xticklabels(), size=15)
        plt.setp(ax.get_yticklabels(), size=15)
    else:
        ax.set_ylabel('Frequency [Hz]', fontsize=16)
        ax.set_ylim([minfreq, maxfreq])
        plt.setp(ax.get_xticklabels(), size=15)
        plt.setp(ax.get_yticklabels(), size=15)

    # for colorbar
    cb = plt.colorbar(psd_das, ax=ax, cax=cax)
    cb_label = 'Amp'
    cb.ax.set_ylabel(cb_label, fontsize=13)
    tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar ticks number
    cb.locator = tick_locator
    cb.ax.tick_params(labelsize=12.5)
    cb.ax.ticklabel_format(style='sci', scilimits=(-1, 1))
    # cb.ax.set_title('unit', fontsize=12.5)
    cb.update_ticks()

    if times[-1] - times[0] > 300:
        # ax.ylabel('Time')
        ax.set_xticks(choose_num[1:])
        ax.set_xticklabels(time_plot_xticks[1:],
                           horizontalalignment='right', rotation=30, fontsize=13)
    else:
        ax.set_xlabel('Time [s]', fontsize=16)

    if show:
        plt.show()
    elif figsave:
        plt.savefig(figsave)
        plt.clf()
        plt.close()

def plot_psd_freq_time_single(psds_single, times, minfreq, maxfreq, channel,
                       figsave=None, show=False, timeticks=10, figsize=None, title=None, dpi=300):
    '''
    Plot time psd of single channel in some freq bands
    By Hardy Hong 2024.01.17

    :param psds_single:
    :param times:
    :param minfreq:
    :param maxfreq:
    :param channel:
    :param cmap:
    :param norm:
    :param smooth:
    :param figsave:
    :param show:
    :param timeticks:
    :param figsize:
    :param title:
    :param dpi:
    :param scale:
    :return:
    '''

    if figsize is None:
        if times[-1] - times[0] > 300:
            figsize = [8.5, 6.5]
        else:
            figsize = [8.5, 5.5]

    time_plot_xticks = []
    nums = int(len(times) / timeticks)
    choose_num = np.arange(20) * nums
    k = 0
    if times[-1] - times[0] > 300:
        for i in choose_num:
            if i < len(times):
                if i == 1:
                    str_time_xtick = times[i].strftime("%Y-%m-%dT%H:%M:%S")
                else:
                    str_time_xtick = times[i].strftime("%d..%H:%M:%S")
                time_plot_xticks.append(str_time_xtick)
                k += 1
            else:
                break
        choose_num = choose_num[0:k]

    dt = times[1] - times[0]
    times_s = np.arange(len(times)) * dt + dt/2

    # plot psd
    fig = plt.figure(figsize=figsize, dpi=dpi)
    if times[-1] - times[0] > 300:
        ax = plt.axes([0.125, 0.16, 0.85, 0.75])
        for i in range(psds_single.shape[1]):
            legends = str(minfreq[i]) + '-' + str(maxfreq[i]) + ' Hz'
            ax.plot(np.arange(len(times_s)), psds_single[:, i], '-', linewidth=1.5, label=legends)
    else:
        ax = plt.axes([0.125, 0.12, 0.85, 0.80])
        for i in range(psds_single.shape[1]):
            legends = str(minfreq[i]) + '-' + str(maxfreq[i]) + ' Hz'
            ax.plot(times_s, psds_single[:, i], '-', linewidth=1.5, label=legends)


    if title:
        title = title + ' PSD' + ' | ' + times[0].strftime("%Y-%m-%dT%H:%M:%S") + ' | ' \
                + 'Channel-' + str(channel)
    else:
        title = 'PSD' + ' | ' + times[0].strftime("%Y-%m-%dT%H:%M:%S") + ' | ' \
                + 'Channel-' + str(channel)
    ax.set_title(title, fontsize=17)
    # ylabes = "Amp [StrainRate$^{2} + r"//"
    ax.set_ylabel('Amp [$StrainRate^{2}/Hz$] [dB]', fontsize=15)
    plt.setp(ax.get_xticklabels(), size=14)
    plt.setp(ax.get_yticklabels(), size=14)
    ax.grid(linestyle='--')
    ax.legend()

    if times[-1] - times[0] > 300:
        # ax.ylabel('Time')
        ax.set_xticks(choose_num)
        ax.set_xticklabels(time_plot_xticks,
                           horizontalalignment='right', rotation=30, fontsize=12)
    else:
        ax.set_xlabel('Time [s]', fontsize=15)

    if show:
        plt.show()
    elif figsave:
        plt.savefig(figsave)
        plt.clf()
        plt.close()

def plot_psd_channel_statistic(fss_channel, fss_count, psds_channel, times, channel, period=False, segment=None,
                       figsave=None, show=False, figsize=None, title=None, dpi=300, bins=50,
                    cmap=None, types='pqlx', norm=None, freq_log=False, smooth=False):
    '''
    Plot the 2D histgram of one channel PSD
    By Hardy Hong 2024.01.18

    :param psds_channel:
    :param times:
    :param minfreq:
    :param maxfreq:
    :param channel:
    :param figsave:
    :param show:
    :param figsize:
    :param title:
    :param dpi:
    :return:
    '''
    from obspy.imaging.cm import pqlx
    from matplotlib.ticker import LogLocator
    if cmap is None:
        cmap = pqlx

    if figsize is None:
        figsize = [8, 5.5]

    # if period:
    #     fss_channel = 1/fss_channel

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes([0.120, 0.12, 0.762, 0.80])
    cax = fig.add_axes([0.902, 0.12, 0.01, ax.get_position().height])
    if types == 'seg':
        fss_channel = np.reshape(fss_channel, [len(times), fss_count]).T
        psds_channel = np.reshape(psds_channel, [len(times), fss_count]).T
        hh_all = np.zeros([bins, bins])
        for i in range(len(times)):
            hh, xedges, yedges = np.histogram2d(psds_channel[:, i], fss_channel[:, i], bins=bins)
            hh = np.where(hh > 1, 1, hh)
            hh_all += hh
        hh_all = hh_all / len(times) * 100       # percent
    else:
        hh, xedges, yedges = np.histogram2d(psds_channel, fss_channel, bins=bins)
        hh_freq = np.sum(hh, axis=0)
        hh_all = hh / hh_freq * 100
    if norm is None:
        norm = [0, np.max(hh_all)]

    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    xlims = [yedges[0], yedges[-1]]
    if smooth:
        psd_das = ax.imshow(hh_all, cmap=cmap, extent=extent, aspect='auto', vmin=norm[0], vmax=norm[1],
                            interpolation='hanning')
    else:
        psd_das = ax.imshow(hh_all, cmap=cmap, extent=extent, aspect='auto', vmin=norm[0], vmax=norm[1])

    if title:
        title = title + ' PSD' + ' | ' + times[0].strftime("%Y-%m-%dT%H:%M:%S") + '......' + \
                times[-1].strftime("%Y-%m-%dT%H:%M:%S") + ' | ' + str(len(times)) + 'Segments'
    else:
        title = 'PSD' + ' | ' + times[0].strftime("%Y-%m-%dT%H:%M:%S") + '......' +\
                times[-1].strftime("%Y-%m-%dT%H:%M:%S") + ' | ' + str(len(times)) + '-Segments'
    ax.set_title(title, fontsize=15)
    ax.set_ylabel('Amp [$StrainRate^{2}/Hz$] [dB]', fontsize=15)
    if period:
        # ax.set_xscale('log')
        ax.set_xlabel('Period [s]', fontsize=15)
        ticks = ax.set_xticks([-2, -1, 0, 1, 2, 3])
        labels = ax.set_xticklabels(['0.01', '0.1', '1', '10', '100', '1000'])
        minor_ticks = np.log10((np.arange(8)+1) * 0.01 + 0.01)
        for i in range(5):
            minor_ticks_1 = np.log10((np.arange(8) + 1) * 0.1 * 10**i + 0.1 * 10**i)
            minor_ticks = np.hstack((minor_ticks, minor_ticks_1))
        ax.set_xticks(minor_ticks, minor=True)
    else:
        ax.set_xlabel('Frequency [Hz]', fontsize=15)
        if freq_log:
            ax.set_xscale('log')
    ax.set_xlim(xlims)
    ax.set_ylim([xedges[0], xedges[-1]])
    ax.grid(linestyle=':', which='both')
    plt.setp(ax.get_xticklabels(), size=14)
    plt.setp(ax.get_yticklabels(), size=14)
    dt = times[1] - times[0]
    if period:
        x_loc = yedges[0] + (yedges[-1] - yedges[0]) / 50
    else:
        x_loc = yedges[0] + (yedges[-1] - yedges[0]) / 50
    ax.text(x=x_loc,
            y=(xedges[-1] - xedges[0]) / 20 + xedges[0],
            s='Channel:'+str(channel)+'   '+'Seg:'+str(segment)+'s   '+'Step:'+str(dt)+'s',
            fontdict=dict(fontsize=8, color='k', family='monospace'),
            # 添加文字背景色
            bbox={'facecolor': 'w',
                  'edgecolor': 'k',
                  'alpha': 0.25,  # box alpha
                  'pad': 0.5,  # the dist from font to box
                  'boxstyle': 'round'})

    # for colorbar
    cb = plt.colorbar(psd_das, ax=ax, cax=cax)
    cb_label = '[%]'
    cb.ax.set_ylabel(cb_label, fontsize=12)
    tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar ticks number
    cb.locator = tick_locator
    cb.ax.tick_params(labelsize=11.5)
    # cb.ax.ticklabel_format(style='sci', scilimits=(-1, 1))
    # cb.ax.set_title('unit', fontsize=12.5)
    cb.update_ticks()

    if show:
        plt.show()
    elif figsave:
        plt.savefig(figsave)
        plt.clf()
        plt.close()
