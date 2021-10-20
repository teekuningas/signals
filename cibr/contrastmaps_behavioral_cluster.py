PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('figure', max_open_warning=0)
    matplotlib.use('Agg')

import pyface.qt

import sys
import argparse
import os

import mne
import scipy
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import matplotlib as mpl
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats

from signals.cibr.lib.stc import plot_vol_stc_brainmap
from signals.cibr.lib.sensor import plot_sensor_topomap

from signals.cibr.lib.utils import MidpointNormalize

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path')
    parser.add_argument('--questionnaire')
    parser.add_argument('--identifier')
    parser.add_argument('--drop')
    parser.add_argument('--spacing')
    parser.add_argument('--example_raw')
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

    # read behavioral questionnaire data
    questionnaire = []
    with open(cli_args.questionnaire, 'r') as f:
        lines = f.readlines()
        header = (['id'] + [elem.strip('\n') for elem in 
                           lines[0][1:].split(',')])
        for line in lines[1:]:
            elems = [elem.strip('\n') for elem in line.split(',')]
            questionnaire.append([elems[0].zfill(3)] + elems[1:])

    questionnaire = pd.DataFrame(questionnaire, columns=header)

    behav_measure = 'BAI'

    behavs = []
    for name in names:
        behavs.append(
            float(questionnaire[questionnaire['id'] == name][behav_measure].values[0]))


    # corr_fun = lambda a, b: scipy.stats.spearmanr(a, b)[0]
    corr_fun = lambda a, b: scipy.stats.pearsonr(a, b)[0]

    # weights = np.cbrt(np.abs(np.mean(norm_data, axis=0)))
    weights = np.sqrt(np.abs(np.mean(norm_data, axis=0)))
    # weights = weights / np.max(weights)

    # TRY WITH ZERO WEIGHTS ( BAI )
    # weights = np.ones(np.shape(norm_data)[1])


    statmap = []
    for vert_idx in range(data.shape[1]):
        Y = data[:, vert_idx]
        X = behavs
        corrcoef = corr_fun(X, Y)
        statmap.append(corrcoef)

    statmap = statmap * weights

    if len(statmap) > 500:
        vertices = np.array([int(vx) for vx in vertex_list[0]])

    crit_val = np.percentile(np.abs(statmap), 97.5)
    print("Critical value: " + str(crit_val))

    def stat_fun(x):
        corrcoefs = np.array([corr_fun(x[:, idx], behavs) * weights[idx] 
                              for idx in range(x.shape[1])])
        return corrcoefs

    import scipy.sparse
    connectivity = scipy.sparse.coo_matrix(np.corrcoef(norm_data.T))
    results = mne.stats.cluster_level.spatio_temporal_cluster_1samp_test(
        data[:, np.newaxis, :],
        connectivity=connectivity,
        n_permutations=1024,
        out_type='indices',
        threshold=crit_val,
        stat_fun=stat_fun)

    clusters = []
    for cluster_idx in range(len(results[2])):

        pvalue = results[2][cluster_idx]
        # if pvalue > 0.05:
        #     continue

        # add jitter to avoid overlay bug
        cluster_map = np.array([np.abs(np.random.normal(scale=0.001)) 
                                for _ in range(data.shape[1])])

        for vert_idx in results[1][cluster_idx][1]:
            cluster_map[vert_idx] = 1 + np.random.normal(scale=0.001)

        clusters.append((pvalue, cluster_map))

    n_clusters = len(clusters)

    fig = plt.figure()
    plt.rcParams.update({'font.size': 50.0})
    fig.set_size_inches(70, 10)
    fig_dpi = 20

    ax_stats = plt.subplot2grid((5, 60), (0, 0), rowspan=5, colspan=18)
    ax_stats_cbar = plt.subplot2grid((5, 60), (1, 19), rowspan=3, colspan=1)

    ax_clusters = []
    for cluster_idx in range(n_clusters):
        ax_clusters.append(
            plt.subplot2grid((5, 60), (0, 24 + cluster_idx*18), rowspan=5, colspan=18))


    if len(statmap) > 500:
        plot_vol_stc_brainmap(statmap, vertices, vol_spacing, subjects_dir,
                          ax_stats, cap=0.9)
        for cluster_idx in range(n_clusters):
            plot_vol_stc_brainmap(clusters[cluster_idx][1], vertices,
                                  vol_spacing, subjects_dir, ax_clusters[cluster_idx],
                                  cmap='PiYG', cap=0.0)
    else:
        plot_sensor_topomap(statmap, raw.info, ax_stats)

        for cluster_idx in range(n_clusters):
            plot_sensor_topomap(clusters[cluster_idx][1], raw.info,
                               ax_clusters[cluster_idx], cmap='PiYG')

    cmap = mpl.cm.RdBu_r

    norm = MidpointNormalize(np.min(statmap), np.max(statmap))
    cb = mpl.colorbar.ColorbarBase(ax_stats_cbar, cmap=cmap,
        norm=norm,
        orientation='vertical')
    cb.set_label('Correlation coefficient', labelpad=12)

    for idx, ax in enumerate(ax_clusters):
        title = 'Cluster (p-value {1:.2g})'.format(idx+1, clusters[idx][0])
        ax.set_title(title)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fname = cli_args.identifier + '.png'

        fig.savefig(os.path.join(save_path, fname), dpi=fig_dpi)

