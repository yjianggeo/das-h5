"""Plot Poincare spectral-coherence results: coherence map, BI profile and
band-pair score heatmap."""

import numpy as np
from matplotlib import pyplot as plt


def plot_coherence_map(c_smooth, wind_step=1.0, channel_begin=0, block=10,
                       block_step=None, bands=None, title=None,
                       figsave=None, show=False, dpi=300, figsize=None,
                       cmap='inferno', vmax_quantile=99.0):
    '''
    Plot the smoothed normalised coherence map C_smooth (time-channel).
    By yjianggeo 2026.09.02

    :param c_smooth:        smoothed coherence map [nwin, nblk]
    :param wind_step:       window hop [s]
    :param channel_begin:   first channel of the data
    :param block:           channels per block M
    :param block_step:      block step in channels (default = block)
    :param bands:           ((l0, l1), (h0, h1)) used for the fingerprints
    :param title:           head of title
    :param figsave:         save fig file
    :param show:            whether to show
    :param dpi:             dpi
    :param figsize:         figsize
    :param cmap:            colormap
    :param vmax_quantile:   colorbar upper quantile [%]
    :return:                None
    '''
    if block_step is None:
        block_step = block
    if figsize is None:
        figsize = [10, 5.5]
    nwin, nblk = c_smooth.shape
    ch0 = channel_begin + block / 2.0
    ch1 = channel_begin + (nblk - 1) * block_step + block / 2.0
    vmax = np.percentile(c_smooth, vmax_quantile)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes([0.10, 0.12, 0.80, 0.80])
    cax = fig.add_axes([0.915, 0.12, 0.015, 0.80])
    im = ax.imshow(c_smooth.T, cmap=cmap, aspect='auto', origin='lower',
                   extent=[0, nwin * wind_step, ch0, ch1], vmin=0, vmax=vmax)
    band_str = ''
    if bands:
        band_str = ' | %g-%g / %g-%g Hz' % (bands[0][0], bands[0][1],
                                            bands[1][0], bands[1][1])
    if title:
        title = title + ' C_smooth' + band_str
    else:
        title = 'Poincare spectral coherence C_smooth' + band_str
    ax.set_title(title, fontsize=15)
    ax.set_xlabel('Time [s]', fontsize=13)
    ax.set_ylabel('Channel [number]', fontsize=13)
    plt.setp(ax.get_xticklabels(), size=11)
    plt.setp(ax.get_yticklabels(), size=11)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('C_smooth', fontsize=12)
    if figsave:
        plt.savefig(figsave, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_bi_profile(bi, channel_begin=0, block=10, block_step=None,
                    bands=None, title=None, figsave=None,
                    show=False, dpi=300, figsize=None, log_scale=True):
    '''
    Plot the BI (burial / coupling-irregularity index) profile along the
    channels.
    By yjianggeo 2026.09.02

    :param bi:              BI profile [nblk]
    :param channel_begin:   first channel of the data
    :param block:           channels per block M
    :param block_step:      block step in channels (default = block)
    :param bands:           ((l0, l1), (h0, h1)) used for the fingerprints
    :param title:           head of title
    :param figsave:         save fig file
    :param show:            whether to show
    :param dpi:             dpi
    :param figsize:         figsize
    :param log_scale:       if true, plot BI on a log y-axis
    :return:                None
    '''
    if block_step is None:
        block_step = block
    if figsize is None:
        figsize = [10, 5]
    channels = channel_begin + np.arange(len(bi)) * block_step + block / 2.0

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes([0.10, 0.13, 0.85, 0.79])
    ax.plot(channels, bi, lw=1.6, marker='o', ms=3.5, color='tab:blue')
    if log_scale:
        ax.set_yscale('log')
    band_str = ''
    if bands:
        band_str = ' | %g-%g / %g-%g Hz' % (bands[0][0], bands[0][1],
                                            bands[1][0], bands[1][1])
    if title:
        title = title + ' BI' + band_str
    else:
        title = 'BI profile (median of C_smooth over time)' + band_str
    ax.set_title(title, fontsize=15)
    ax.set_xlabel('Channel [number]', fontsize=13)
    ax.set_ylabel('BI', fontsize=13)
    ax.grid(alpha=0.3, which='both')
    plt.setp(ax.get_xticklabels(), size=11)
    plt.setp(ax.get_yticklabels(), size=11)
    if figsave:
        plt.savefig(figsave, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_band_scores(scores, figsave=None, show=False, dpi=300, figsize=None):
    '''
    Plot the band-pair scan scores as a low-band x high-band heatmap.
    By yjianggeo 2026.09.02

    :param scores:  rows (l0, l1, h0, h1, contrast, stability, score)
                    from optimize_band_pair
    :param figsave: save fig file
    :param show:    whether to show
    :param dpi:     dpi
    :param figsize: figsize
    :return:        None
    '''
    if figsize is None:
        figsize = [11, 4.5]
    lows = sorted({(r[0], r[1]) for r in scores})
    highs = sorted({(r[2], r[3]) for r in scores})
    grid = np.full((len(lows), len(highs)), np.nan)
    for r in scores:
        grid[lows.index((r[0], r[1])), highs.index((r[2], r[3]))] = r[6]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes([0.08, 0.22, 0.83, 0.70])
    cax = fig.add_axes([0.925, 0.22, 0.015, 0.70])
    im = ax.imshow(grid, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xticks(range(len(highs)))
    ax.set_xticklabels(['%g-%g' % h for h in highs], rotation=90, fontsize=6)
    ax.set_yticks(range(len(lows)))
    ax.set_yticklabels(['%g-%g' % l for l in lows], fontsize=8)
    ax.set_xlabel('High band [Hz]', fontsize=12)
    ax.set_ylabel('Low band [Hz]', fontsize=12)
    ax.set_title('Band-pair score = spatial contrast x split-half stability',
                 fontsize=13)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('score', fontsize=11)
    if figsave:
        plt.savefig(figsave, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
