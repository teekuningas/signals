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
import matplotlib as mpl


from signals.cibr.lib.stc import plot_vol_stc_brainmap
from signals.cibr.lib.stc import get_vol_labeled_data
from signals.cibr.lib.stc import plot_vol_stc_labels
from signals.cibr.lib.sensor import plot_sensor_topomap

from signals.cibr.lib.utils import MidpointNormalize

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import logging
logger = logging.getLogger("mne")
logger.setLevel(logging.ERROR)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path')
    parser.add_argument('--identifier')
    parser.add_argument('--example_raw')
    parser.add_argument('--drop')
    parser.add_argument('--spacing')
    parser.add_argument('--coefficients_1', nargs='+')
    parser.add_argument('--coefficients_2', nargs='+')
    parser.add_argument('--coefficients_norm', nargs='+')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    vol_spacing = '10'
    if cli_args.spacing is not None:
        vol_spacing = str(cli_args.spacing)

    data_1 = []
    data_2 = []
    norm_data = []
    names_1 = []
    names_2 = []
    names_norm = []
    vertex_list = []

    # meditaatio
    def name_from_fname(fname):
        return fname.split('/')[-1].split('_')[1]

    # # fdmsa
    # def name_from_fname(fname):
    #     if 'heart' in fname and 'note' in fname:
    #         return '_'.join(fname.split('/')[-1].split('_')[1:][:-2])
    #     else:
    #         return '_'.join(fname.split('/')[-1].split('_')[1:][:-1])


    # read contrast data from all participants
    drop_keys = cli_args.drop.split(' ') if cli_args.drop else []

    for fname in sorted(cli_args.coefficients_1):
        if any([key in fname for key in drop_keys]):
            continue

        with open(fname, 'r') as f:
            lines = f.readlines()
            vertex_list.append([val for val in lines[0].strip().split(', ')])
            data_1.append([float(val) for val in lines[1].split(', ')])
            names_1.append(name_from_fname(fname))
    data_1 = np.array(data_1)

    for fname in sorted(cli_args.coefficients_2):
        if any([key in fname for key in drop_keys]):
            continue

        with open(fname, 'r') as f:
            lines = f.readlines()
            data_2.append([float(val) for val in lines[1].split(', ')])
            names_2.append(name_from_fname(fname))
    data_2 = np.array(data_2)

    for fname in sorted(cli_args.coefficients_norm):
        if any([key in fname for key in drop_keys]):
            continue

        with open(fname, 'r') as f:
            lines = f.readlines()
            norm_data.append([float(val) for val in lines[1].split(', ')])
            names_norm.append(name_from_fname(fname))
    norm_data = np.array(norm_data)

    if not (names_1 == names_2 == names_norm):
        raise Exception('Names do not match')

    names = names_1

    data = np.mean([data_1, data_2], axis=0)

    # for plotting
    raw = mne.io.Raw(cli_args.example_raw, preload=True)
    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names'])
                       if idx not in mne.pick_types(raw.info, meg=True)])

    # weights = np.cbrt(np.abs(np.mean(norm_data, axis=0)))
    weights = np.abs(np.mean(norm_data, axis=0))
    # weights = np.ones(data.shape[1])
    weights = weights / np.max(weights)

    meanmap = np.mean(data, axis=0)

    if len(meanmap) > 500:
        vertices = np.array([int(vx) for vx in vertex_list[0]])

    def plot_labeled_data(brainmap, fname):
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        plot_vol_stc_labels(brainmap, vertices, vol_spacing, subjects_dir, ax, n_labels=12)
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            fig.savefig(os.path.join(save_path, fname), dpi=100)

    if len(meanmap) > 500:
        plot_labeled_data(meanmap, cli_args.identifier + '_mean_labels.png')

    statmap = []

    for spatial_idx in range(data.shape[1]):
        stat = scipy.stats.ttest_1samp(data[:, spatial_idx], 0)[0]
        statmap.append(stat)

    statmap = statmap * weights

    if len(meanmap) > 500:
        plot_labeled_data(statmap, cli_args.identifier + '_stat_labels.png')

    # crit_val = scipy.stats.t.ppf(0.995, df=len(statmap))
    crit_val = np.abs(np.percentile(np.abs(statmap), 95.0))
    print("Critical value: " + str(crit_val))

    def stat_fun(x):
        stats = []
        for idx in range(x.shape[1]):

            stat = scipy.stats.ttest_1samp(x[:, idx], 0)[0] * weights[idx]

            stats.append(stat)
        return np.array(stats)
 
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
        cluster_map = np.array([np.abs(np.random.normal(scale=0.001)) for _ in 
                                range(data.shape[1])])

        for vert_idx in results[1][cluster_idx][1]:
            cluster_map[vert_idx] = 1 + np.random.normal(scale=0.001)

        if len(meanmap) > 500:
            plot_labeled_data(cluster_map, cli_args.identifier + '_cluster_' + str(cluster_idx+1) + '_labels.png')

        pvalue = results[2][cluster_idx]

        clusters.append((pvalue, cluster_map))

    # hack to get same-sized images..
    if len(clusters) == 1:
        clusters = [clusters[0], clusters[0]]

    n_clusters = len(clusters)

    # finally I know how to set these. 
    # to get high accuracy vol plots, 
    # set inches high and dpi low.
    fig = plt.figure()
    fig.suptitle(cli_args.identifier, fontsize=50.0)

    plt.rcParams.update({'font.size': 25.0})
    fig.set_size_inches(20, 40)
    fig_dpi = 100

    ax_mean = plt.subplot2grid((2*(2 + n_clusters), 20), (0, 0), colspan=19)
    ax_mean_cbar = plt.subplot2grid((2*(2 + n_clusters), 20), (0, 19))
    ax_stats = plt.subplot2grid((2*(2 + n_clusters), 20), (2, 0), colspan=19)
    ax_stats_cbar = plt.subplot2grid((2*(2 + n_clusters), 20), (2, 19))

    ax_clusters = []
    for cluster_idx in range(n_clusters):
        ax_clusters.append(
            plt.subplot2grid((2*(2 + n_clusters), 20), (4 + cluster_idx * 2, 0), colspan=19))


    if len(meanmap) > 500:
        plot_vol_stc_brainmap(statmap, vertices, vol_spacing, subjects_dir,
                          ax_stats, cap=0.9)
        plot_vol_stc_brainmap(meanmap, vertices, vol_spacing, subjects_dir,
                          ax_mean, cap=0.9)
        for cluster_idx in range(n_clusters):
            plot_vol_stc_brainmap(clusters[cluster_idx][1], vertices,
                                  vol_spacing, subjects_dir, ax_clusters[cluster_idx],
                                  cmap='PiYG', cap=0.0)

    else:
        plot_sensor_topomap(statmap, raw.info, ax_stats)
        plot_sensor_topomap(meanmap, raw.info, ax_mean)

        for cluster_idx in range(n_clusters):
            plot_sensor_topomap(clusters[cluster_idx][1], raw.info,
                               ax_clusters[cluster_idx], cmap='PiYG')

    cmap = mpl.cm.RdBu_r

    norm = MidpointNormalize(np.min(statmap), np.max(statmap))
    cb = mpl.colorbar.ColorbarBase(ax_stats_cbar, cmap=cmap,
        norm=norm,
        orientation='vertical')
    cb.set_label('Normalized t-value', labelpad=10)

    norm = MidpointNormalize(np.min(meanmap), np.max(meanmap))
    cb = mpl.colorbar.ColorbarBase(ax_mean_cbar, cmap=cmap,
        norm=norm,
        orientation='vertical')
    cb.set_label('MNE current power', labelpad=10)

    ax_mean.set_title('Average over participant-specific spatial task-contrast maps')
    ax_stats.set_title('T-value map')

    for idx, ax in enumerate(ax_clusters):
        title = 'Cluster {0} (p-value {1:.2g})'.format(idx+1, clusters[idx][0])
        ax.set_title(title)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fname = cli_args.identifier + '.png'

        fig.savefig(os.path.join(save_path, fname), dpi=fig_dpi)

