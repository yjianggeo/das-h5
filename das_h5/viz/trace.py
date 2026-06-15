"""Single-trace plotting and spectrograms for DAS channels.

Repackaged from DASh5 das_single.py (By Hardy Hong). Plotting unchanged.
"""

import os
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib.colors import Normalize
from obspy.imaging.cm import obspy_sequential

def _nearest_pow_2(x):
    """
    Find power of two nearest to x

    >>> _nearest_pow_2(3)
    2.0
    >>> _nearest_pow_2(15)
    16.0

    :type x: float
    :param x: Number
    :rtype: int
    :return: Nearest power of 2 to x
    """
    a = math.pow(2, math.ceil(np.log2(x)))
    b = math.pow(2, math.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b


def spectrograms(data, samp_rate, per_lap=0.9, wlen=None, log=False,
                 outfile=None, fmt=None, axes=None, dbscale=False,
                 mult=8.0, cmap=obspy_sequential, zorder=None, title=None,
                 show=True, clip=(0.0, 1.0), values_back=False, time_begin=0,
                 norms=None):
    """
    (Little correction from the obspy.imagine.spectrograms
    By Hardy Hong 2023.07.12)
    Computes and plots spectrogram of the input data.

    :param data: Input data
    :type samp_rate: float
    :param samp_rate: Samplerate in Hz
    :type per_lap: float
    :param per_lap: Percentage of overlap of sliding window, ranging from 0
        to 1. High overlaps take a long time to compute.
    :type wlen: int or float
    :param wlen: Window length for fft in seconds. If this parameter is too
        small, the calculation will take forever. If None, it defaults to a
        window length matching 128 samples.
    :type log: bool
    :param log: Logarithmic frequency axis if True, linear frequency axis
        otherwise.
    :type outfile: str
    :param outfile: String for the filename of output file, if None
        interactive plotting is activated.
    :type fmt: str
    :param fmt: Format of image to save
    :type axes: :class:`matplotlib.axes.Axes`
    :param axes: Plot into given axes, this deactivates the fmt and
        outfile option.
    :type dbscale: bool
    :param dbscale: If True 10 * log10 of color values is taken, if False the
        sqrt is taken.
    :type mult: float
    :param mult: Pad zeros to length mult * wlen. This will make the
        spectrogram smoother.
    :type cmap: :class:`matplotlib.colors.Colormap`
    :param cmap: Specify a custom colormap instance. If not specified, then the
        default ObsPy sequential colormap is used.
    :type zorder: float
    :param zorder: Specify the zorder of the plot. Only of importance if other
        plots in the same axes are executed.
    :type title: str
    :param title: Set the plot title
    :type show: bool
    :param show: Do not call `plt.show()` at end of routine. That way, further
        modifications can be done to the figure before showing it.
    :type clip: [float, float]
    :param clip: adjust colormap to clip at lower and/or upper end. The given
        percentages of the amplitude range (linear or logarithmic depending
        on option `dbscale`) are clipped.
    :type values_back: True of False
    :param values_back: Return values of freq, time, specgram
    :type time_begin： float
    :param time_begin: b of data
    :type norms： float
    :param norms: plot norm of imshow
    """

    # enforce float for samp_rate
    samp_rate = float(samp_rate)

    # set wlen from samp_rate if not specified otherwise
    if not wlen:
        wlen = 128 / samp_rate

    npts = len(data)

    # nfft needs to be an integer, otherwise a deprecation will be raised
    # XXX add condition for too many windows => calculation takes for ever
    nfft = int(_nearest_pow_2(wlen * samp_rate))

    if npts < nfft:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, sampling rate '
               f'{samp_rate} Hz)')
        raise ValueError(msg)

    if mult is not None:
        mult = int(_nearest_pow_2(mult))
        mult = mult * nfft
    nlap = int(nfft * float(per_lap))

    data = data - data.mean()
    end = npts / samp_rate

    # Here we call not plt.specgram as this already produces a plot
    # matplotlib.mlab.specgram should be faster as it computes only the
    # arrays
    # XXX mlab.specgram uses fft, would be better and faster use rfft
    specgram, freq, time = mlab.specgram(data, Fs=samp_rate, NFFT=nfft,
                                         pad_to=mult, noverlap=nlap)

    if len(time) < 2:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, {nlap} samples window '
               f'overlap, sampling rate {samp_rate} Hz)')
        raise ValueError(msg)

    # db scale and remove zero/offset for amplitude
    if dbscale:
        specgram = 10 * np.log10(specgram[1:, :])
    else:
        specgram = np.sqrt(specgram[1:, :])
    freq = freq[1:]

    vmin, vmax = clip
    if vmin < 0 or vmax > 1 or vmin >= vmax:
        msg = "Invalid parameters for clip option."
        raise ValueError(msg)
    _range = float(specgram.max() - specgram.min())
    vmin = specgram.min() + vmin * _range
    vmax = specgram.min() + vmax * _range
    norm = Normalize(vmin, vmax, clip=True)

    if not axes:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = axes

    # calculate half bin width
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0

    # argument None is not allowed for kwargs on matplotlib python 3.3
    kwargs = {k: v for k, v in (('cmap', cmap), ('zorder', zorder))
              if v is not None}

    if log:
        # pcolor expects one bin more at the right end
        freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
        time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
        # center bin
        time -= halfbin_time
        freq -= halfbin_freq
        # Log scaling for frequency values (y-axis)
        ax.set_yscale('log')
        # Plot times
        ax.pcolormesh(time, freq, specgram, norm=norm, **kwargs)
    else:
        # this method is much much faster!
        specgram = np.flipud(specgram)
        # center bin
        extent = (time[0] - halfbin_time + time_begin, time[-1] + halfbin_time + time_begin,
                  freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
        if norms:
            ax.imshow(specgram/vmax, interpolation="nearest", extent=extent, vmin=norms[0], vmax=norms[1], **kwargs)
        else:
            ax.imshow(specgram/vmax, interpolation="nearest", extent=extent, **kwargs)

    # set correct way of axis, whitespace before and after with window
    # length
    ax.axis('tight')
    ax.set_xlim(0, end)
    ax.grid(False)

    if axes:
        return ax

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    if title:
        ax.set_title(title)

    if values_back:
        return specgram, time + time_begin, freq

    if not os.environ.get('SPHINXBUILD'):
        # ignoring all NumPy warnings during plot
        with np.errstate(all='ignore'):
            plt.draw()
    if outfile:
        if fmt:
            fig.savefig(outfile, format=fmt)
            plt.clf()
            plt.close()
        else:
            fig.savefig(outfile)
            plt.clf()
            plt.close()
    elif show:
        plt.show()
    else:
        return fig

def single_plot(data, delta, begin_time=0, end_time=10, title='DAS Trace',
                figsize=(10, 6), dpi=200, norm=(0, 1), outfig=None, show=False, **kwargs):
    '''
    Single Trace plot with amp and spec.
    Little correction form obspy.
    By Hardy Hong 2023.07.12


    :param data:         1d-data array
    :param delta:        data delta
    :param begin_time:   data relative origin time for the ref-time
    :param end_time:     data relative end time for the ref-time
                            (>begin_time, but may not equal true end time
                            when the time-lim has been given)
    :param title:         the fig title
    :param figsize:       fig-size
    :param dpi:           fig dpi
    :param norm           spectrogram data norm
    :param outfig:        out fig name
    :param show:          if true, fig shows
    :param kwargs:        time_lim (plot xlim)
                          spec_lim (plot spec ylim)
                          rate (show scale of wave and spec)
    :return:              fig or others
    '''
    spe_plot_min, spe_plot_max = kwargs.get('spec_lim', (0, 1 / delta * 0.5))
    time_b, time_e = kwargs.get('time_lim', (begin_time, end_time))
    rate = kwargs.get('show_rate', 0.25/0.8)
    xlim = [time_b, time_e]
    ylim = [spe_plot_min, spe_plot_max]


    data_end_time = len(data) * delta + begin_time
    time_s = np.arange(begin_time, data_end_time, delta)
    times = time_s[0: len(data)]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax1 = plt.axes([0.10, 0.1 + 0.8 * (1 - rate), 0.85, rate * 0.8])
    base = ax1.plot(times, data, '-', c='#303030')
    ax1.set_xlim(xlim)
    ax1.set_xticks([])
    plt.title(title, y=1.1, fontsize=16)
    ax1.tick_params(labelsize=13)
    ax1.set_ylabel('Amp [s$^{-1}$]', fontsize=14)
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)
    ax1.spines['top'].set_linewidth(1)

    # plot single trace spectrogram
    ax2 = plt.axes([0.10, 0.10, 0.85, 0.8 * (1 - rate) - 0.001])
    spectrograms(data, samp_rate=1/delta, axes=ax2, norms=norm,
                 wlen=(data_end_time - begin_time) * 0.025, time_begin=begin_time)
    # ax2.set_title(idas_title, y=1.07)
    ax2.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.set_ylabel('Frequency [Hz]', fontsize=14)
    ax2.set_xlabel('Time [s]', fontsize=14)
    ax2.tick_params(labelsize=14)
    ax2.spines['bottom'].set_linewidth(1)
    ax2.spines['left'].set_linewidth(1)
    ax2.spines['right'].set_linewidth(1)
    ax2.spines['top'].set_linewidth(1)
    # iddi = st_idas_cp[single_num].id
    # ax2.text(x=t_start + (t_end - t_start) / 25,  # 文本x轴坐标
    #          y=spe_plot_max - (spe_plot_max - spe_plot_min) / 10,  # 文本y轴坐标
    #          s=iddi[0:16],  # 文本内容
    #          fontdict=dict(fontsize=8, color='w', family='monospace'),  # 字体属性字典
    #          # 添加文字背景色
    #          bbox={'facecolor': '#000000',  # 填充色
    #                'edgecolor': 'w',  # 外框色
    #                'alpha': 0.25,  # 框透明度
    #                'pad': 0.5,  # 本文与框周围距离
    #                'boxstyle': 'round'})
    if outfig:
        plt.savefig(outfig)
        plt.clf()
        plt.close()
    elif show:
        plt.show()
    else:
        return fig

def single_plot_sac(data, delta, begin_time=0, end_time=10, title='DAS Trace',
                figsize=(10, 6), dpi=200, norm=(0, 1), outfig=None, show=False, **kwargs):
    '''
    Single Trace plot with amp and spec.
    Little correction form obspy.
    By Hardy Hong 2023.07.12


    :param data:         1d-data array
    :param delta:        data delta
    :param begin_time:   data relative origin time for the ref-time
    :param end_time:     data relative end time for the ref-time
                            (>begin_time, but may not equal true end time
                            when the time-lim has been given)
    :param title:         the fig title
    :param figsize:       fig-size
    :param dpi:           fig dpi
    :param norm           spectrogram data norm
    :param outfig:        out fig name
    :param show:          if true, fig shows
    :param kwargs:        time_lim (plot xlim)
                          spec_lim (plot spec ylim)
                          rate (show scale of wave and spec)
    :return:              fig or others
    '''
    spe_plot_min, spe_plot_max = kwargs.get('spec_lim', (0, 1 / delta * 0.5))
    time_b, time_e = kwargs.get('time_lim', (begin_time, end_time))
    rate = kwargs.get('show_rate', 0.25/0.8)
    xlim = [time_b, time_e]
    ylim = [spe_plot_min, spe_plot_max]


    data_end_time = len(data) * delta + begin_time
    time_s = np.arange(begin_time, data_end_time, delta)
    times = time_s[0: len(data)]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax1 = plt.axes([0.10, 0.1 + 0.8 * (1 - rate), 0.85, rate * 0.8])
    base = ax1.plot(times, data, '-', c='#303030')
    ax1.set_xlim(xlim)
    ax1.set_xticks([])
    plt.title(title, y=1.1, fontsize=16)
    ax1.tick_params(labelsize=13)
    ax1.set_ylabel('Amp [nm/s]', fontsize=14)
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)
    ax1.spines['top'].set_linewidth(1)

    # plot single trace spectrogram
    ax2 = plt.axes([0.10, 0.10, 0.85, 0.8 * (1 - rate) - 0.001])
    spectrograms(data, samp_rate=1/delta, axes=ax2, norms=norm,
                 wlen=(data_end_time - begin_time) * 0.025, time_begin=begin_time)
    # ax2.set_title(idas_title, y=1.07)
    ax2.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.set_ylabel('Frequency [Hz]', fontsize=14)
    ax2.set_xlabel('Time [s]', fontsize=14)
    ax2.tick_params(labelsize=14)
    ax2.spines['bottom'].set_linewidth(1)
    ax2.spines['left'].set_linewidth(1)
    ax2.spines['right'].set_linewidth(1)
    ax2.spines['top'].set_linewidth(1)
    # iddi = st_idas_cp[single_num].id
    # ax2.text(x=t_start + (t_end - t_start) / 25,  # 文本x轴坐标
    #          y=spe_plot_max - (spe_plot_max - spe_plot_min) / 10,  # 文本y轴坐标
    #          s=iddi[0:16],  # 文本内容
    #          fontdict=dict(fontsize=8, color='w', family='monospace'),  # 字体属性字典
    #          # 添加文字背景色
    #          bbox={'facecolor': '#000000',  # 填充色
    #                'edgecolor': 'w',  # 外框色
    #                'alpha': 0.25,  # 框透明度
    #                'pad': 0.5,  # 本文与框周围距离
    #                'boxstyle': 'round'})
    if outfig:
        plt.savefig(outfig)
        plt.clf()
        plt.close()
    elif show:
        plt.show()
    else:
        return fig
