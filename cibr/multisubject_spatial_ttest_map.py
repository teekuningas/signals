PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.use('Agg')

import pyface.qt

import argparse
import os

import mne
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt 

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

# plt.rcParams.update({'font.size': 10.0})
# plt.rcParams.update({'lines.linewidth': 3.0})

from signals.cibr.lib.stc import plot_vol_stc_brainmap
from signals.cibr.lib.sensor import plot_sensor_topomap

from signals.cibr.lib.utils import MidpointNormalize

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path')
    parser.add_argument('--identifier')
    parser.add_argument('--example_raw')
    parser.add_argument('--drop')
    parser.add_argument('--coefficients', nargs='+')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    data = []
    vertex_list = []

    # read contrast data from all participants
    drop_keys = cli_args.drop.split(' ') if cli_args.drop else []
    for fname in sorted(cli_args.coefficients):
        if any([key in fname for key in drop_keys]):
            continue

        with open(fname, 'r') as f:
            lines = f.readlines()
            vertex_list.append([val for val in lines[0].strip().split(', ')])
            data.append([float(val) for val in lines[1].split(', ')])


    # for plotting
    raw = mne.io.Raw(cli_args.example_raw, preload=True)
    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names'])
                       if idx not in mne.pick_types(raw.info, meg=True)])

    data = np.array(data)

    meanmap = np.mean(data, axis=0)

    weights = (np.std(data, axis=0) / np.max(np.std(data, axis=0)))**(0.1)

    # t test every spatial unit
    tvaluemap = []
    for spatial_idx in range(data.shape[1]):
        tvaluemap.append(scipy.stats.ttest_1samp(data[:, spatial_idx], 0)[0])
    tvaluemap = np.array(tvaluemap * weights)

    crit_val = scipy.stats.t.ppf(0.975, df=data.shape[0])
    print("Critical value: " + str(crit_val))

    def stat_fun(x):
        tvalues = np.array([scipy.stats.ttest_1samp(x[:, idx], 0)[0] * weights[idx]
                            for idx in range(x.shape[1])])
        return tvalues

    import scipy.sparse
    connectivity = scipy.sparse.coo_matrix(np.corrcoef(data.T))
    results = mne.stats.cluster_level.spatio_temporal_cluster_1samp_test(
        data[:, np.newaxis, :], 
        connectivity=connectivity,
        n_permutations=1024,
        out_type='indices',
        threshold=crit_val,
        stat_fun=stat_fun)

    clusters = []
    for cluster_idx in range(len(results[2])):
        # add jitter to avoid overlay bug
        cluster_map = np.array([np.random.normal(scale=0.001) for _ in range(data.shape[1])])
        for vert_idx in results[1][cluster_idx][1]:
            cluster_map[vert_idx] = 1 + np.random.normal(scale=0.001)

        pvalue = results[2][cluster_idx]

        if pvalue > 0.15:
            continue

        clusters.append((pvalue, cluster_map))

    n_clusters = len(clusters)

    # add colorbar and print to axis
    fig, axes = plt.subplots(nrows=(2+n_clusters), ncols=1)
    ax_mean = axes[0]
    ax_tvalues = axes[1]
    ax_clusters = axes[2:]

    if len(tvaluemap) > 500:
        vertices = np.array([int(vx) for vx in vertex_list[0]])
        plot_vol_stc_brainmap(tvaluemap, vertices, '10', subjects_dir,
                          ax_tvalues)
        plot_vol_stc_brainmap(meanmap, vertices, '10', subjects_dir,
                          ax_mean)
        for cluster_idx in range(n_clusters):
            plot_vol_stc_brainmap(clusters[cluster_idx][1], vertices,
                                  '10', subjects_dir, ax_clusters[cluster_idx],
                                  cmap='PiYG')

    else:
        plot_sensor_topomap(tvaluemap, raw.info, ax_tvalues)
        plot_sensor_topomap(meanmap, raw.info, ax_mean)

        for cluster_idx in range(n_clusters):
            plot_sensor_topomap(clusters[cluster_idx][1], raw.info,
                               ax_clusters[cluster_idx], cmap='PiYG')

    cmap = mpl.cm.RdBu_r

    divider = make_axes_locatable(ax_tvalues)
    ax_cbar = divider.append_axes("right", size="2%", pad=0.0)
    norm = MidpointNormalize(np.min(tvaluemap), np.max(tvaluemap))
    cb = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap,
        norm=norm,
        orientation='vertical')

    divider = make_axes_locatable(ax_mean)
    ax_cbar = divider.append_axes("right", size="2%", pad=0.0)
    norm = MidpointNormalize(np.min(meanmap), np.max(meanmap))
    cb = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap,
        norm=norm,
        orientation='vertical')

    ax_mean.set_title('mean')
    ax_tvalues.set_title('t-values')

    for idx, ax in enumerate(ax_clusters):
        ax.set_title('cluster ' + str(idx+1) + ' (pvalue ' + str(clusters[idx][0])  + ')')

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if cli_args.identifier:
            fname = cli_args.identifier + '.png'
        else:
            fname = 'tvaluemap.png'

        fig.savefig(os.path.join(save_path, fname), dpi=310)

