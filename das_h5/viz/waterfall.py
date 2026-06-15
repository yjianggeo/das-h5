"""DAS waterfall (imshow) plot.

Repackaged from DASh5 das_imshow.py (By Hardy Hong). Plotting unchanged.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def das_imshow(data, begin_time=0, delta=0.001,end_time=10, channel_begin=0, channel_end=30,
               figsize_arr=(10, 7), direction='vertical', dpii=200, data_norm=(-1, 1),
               colormap='gray', title='title', outfig=None, show=False, **kwargs):
    '''
     Plot the imshow fig from h5_data
    By Hardy Hong 2023.07.01

    :param data:  h5_data
    :param begin_time:      relative beginning time
    :param delta:           dt
    :param end_time:        relative end time
    :param channel_begin:   beginning channel num
    :param channel_end:     beginning end num
    :param figsize_arr:     figsize [10, 7]
    :param direction:       fig direction || vertical or horizontal
    :param dpii:            dpi
    :param data_norm:       color bar range
    :param colormap:        colormap
    :param title:           title of fig
    :param outfig:          save fig
    :param show:            show the fig
    :param kwargs:          time_lim:        plot time axes lim -- default == (begin_time, endtime)
                            channel_lim:     plot channel axes lim -- default == (channel_begin, channel_end)
    :return:                fig of das
    '''

    if direction == 'vertical':
        data = data
        extent = (channel_begin - 0.5, channel_end + 1 + 0.5,
                  end_time + delta / 2, begin_time - delta / 2)
        xlabel = 'Channel [number]'
        ylabel = 'Time [s]'
        xmin, xmax = kwargs.get('channel_lim', (channel_begin, channel_end))
        ymin, ymax = kwargs.get('time_lim', (begin_time, end_time))
        xlim = [xmin, xmax]
        ylim = [ymax, ymin]
    else:
        data = data.T
        extent = (begin_time - delta / 2, end_time + delta / 2,
                  channel_end + 1 + 0.5, channel_begin - 0.5)
        ylabel = 'Channel [number]'
        xlabel = 'Time [s]'
        ymin, ymax = kwargs.get('channel_lim', (channel_begin, channel_end))
        xmin, xmax = kwargs.get('time_lim', (begin_time, end_time))
        xlim = [xmin, xmax]
        ylim = [ymin, ymax]

    label_fontdict = {
        'fontsize': 16,
    }

    title_fontdict = {
        'fontsize': 17,
    }
    fig = plt.figure(figsize=(figsize_arr[0], figsize_arr[1]), dpi=dpii)
    ax = plt.axes([0.10, 0.10, 0.85, 0.8]) # ax = fig.add_subplot(111)
    ax.imshow(data, cmap=colormap, extent=extent, vmin=data_norm[0], vmax=data_norm[1],
               interpolation="nearest")
    ax.axis('tight')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(False)
    # ax.tick_params(direction='out', length=6, width=1, colors='k')
    ax.tick_params(labelsize=14)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(xlabel, fontdict=label_fontdict)
    ax.set_ylabel(ylabel, fontdict=label_fontdict)
    ax.set_title(title, fontdict=title_fontdict, y=1.03)
    if outfig:
        fig.savefig(outfig)
        #plt.clf()
    elif show:
        plt.show()
    else:
        return fig
