"""DAS wiggle / fill-between plot.

Repackaged from DASh5 das_fillplot.py (By Hardy Hong). Plotting unchanged.
"""

import functools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm, trange


def das_fillplot(data, begin_time=0, delta=0.001, end_time=10, channel_begin=0, channel_end=30,
                 figsize_arr=(10, 7), direction='vertical', dpii=200, gain=1, linewidth=0.005,
                 linecolor='k', linealpha=0.5, fillcolor=(None, None), fillalpha=1, title=None,
                 outfig=None, show=False, **kwargs):
    '''
    Plot das data with fill_between, which is very slow but fine than imshow.
    By Hardy Hong 2023.07.10

    :param data:            h5_data, which have been preprocessed (time, channel, filter and so on)
    :param begin_time:      relative beginning time
    :param delta:           dt
    :param end_time:        relative end time
    :param channel_begin:   beginning channel num
    :param channel_end:     beginning end num
    :param figsize_arr:     fig_size [10, 7]
    :param direction:       fig direction || vertical or horizontal
    :param dpii:            dpi
    :param gain:            data scale -- data * gain
    :param linewidth:       line plot - line width
    :param linecolor:       line plot - line color
    :param linealpha:       line plot - line alpha
    :param fillcolor:       fillcolor - default ('r', 'b') | red and blue
    :param fillalpha:       fillcolor alpha
    :param title:           fig title
    :param outfig:          save fig
    :param show:            show fig
    :param kwargs:          time_lim:        plot time axes lim -- default == (begin_time, endtime)
                            channel_lim:     plot channel axes lim -- default == (channel_begin, channel_end)
    :return:                fig
    '''

    data = data * gain / 1.5                                  # data scale
    end_time_new = begin_time + data.shape[0] * delta
    times = np.arange(begin_time, end_time_new, delta)[0:data.shape[0]]
    fig = plt.figure(figsize=(figsize_arr[0], figsize_arr[1]), dpi=dpii)
    ax = plt.axes([0.10, 0.10, 0.85, 0.8]) # ax = fig.add_subplot(111)

    fillcolor_pos, fillcolor_neg = fillcolor
    fill_kwargs = {"lw": 0, "alpha": fillalpha}
    grid_color = kwargs.get('grid_color', 'black')
    grid_linewidth = kwargs.get('grid_linewidth', 0.001)
    grid_linestyle = kwargs.get('grid_linestyle', ':')
    if direction == 'vertical':
        xlabel = 'Channel [number]'
        ylabel = 'Time [s]'
        xmin, xmax = kwargs.get('channel_lim', (channel_begin, channel_end))
        ymin, ymax = kwargs.get('time_lim', (begin_time, end_time))
        xlim = [xmin, xmax]
        ylim = [ymin, ymax]
        fillfun = functools.partial(ax.fill_betweenx, **fill_kwargs)
    else:
        ylabel = 'Channel [number]'
        xlabel = 'Time [s]'
        ymin, ymax = kwargs.get('channel_lim', (channel_begin, channel_end))
        xmin, xmax = kwargs.get('time_lim', (begin_time, end_time))
        xlim = [xmin, xmax]
        ylim = [ymin, ymax]
        fillfun = functools.partial(ax.fill_between, interpolate=True, **fill_kwargs)

    label_fontdict = {
        'fontsize': 16,
    }

    title_fontdict = {
        'fontsize': 17,
    }

    lines = []
    # for i in range(channel_begin, channel_end + 1):
    for i in tqdm(range(channel_begin, channel_end + 1), desc='Plotting fill figure process: '):
        data_single = data[:, i - channel_begin] + i
        if direction == 'vertical':
            lines += ax.plot(data_single, times, color=linecolor,
                             linewidth=linewidth, alpha=linealpha)
        else:
            lines += ax.plot(times, data_single, color=linecolor,
                             linewidth=linewidth, alpha=linealpha)
        if fillcolor_pos:
            fillfun(times, data_single, i,
                    where=data_single > i,
                    facecolor=fillcolor_pos)
        if fillcolor_neg:
            fillfun(times, data_single, i,
                    where=data_single < i,
                    facecolor=fillcolor_neg)
    ax.axis('tight')
    ax.grid(False)
    ax.tick_params(direction='out', length=6, width=1, colors='k')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    # ax.grid(
    #     color=grid_color,
    #     linestyle=grid_linestyle,
    #     linewidth=grid_linewidth)
    ax.tick_params(labelsize=14)
    ax.set_xlabel(xlabel, fontdict=label_fontdict)
    ax.set_ylabel(ylabel, fontdict=label_fontdict)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontdict=title_fontdict, y=1.03)
    if direction == 'vertical':
        ax.invert_yaxis()
    if outfig:
        fig.savefig(outfig)
        plt.clf()
        plt.close()
    elif show:
        plt.show()
    else:
        return fig
