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
    # corr_fun = lambda a, b: scipy.stats.pearsonr(a, b)[0]

    # weights = np.cbrt(np.abs(np.mean(norm_data, axis=0)))
    # weights = np.sqrt(np.abs(np.mean(norm_data, axis=0)))
    # weights = weights / np.max(weights)

    weights = np.ones(np.shape(norm_data)[1])

    coef_mean = np.zeros(data.shape[1])
    coef_mean_t = np.zeros(data.shape[1])
    coef_a = np.zeros(data.shape[1])
    coef_b = np.zeros(data.shape[1])
    coef_a_t = np.zeros(data.shape[1])
    coef_b_t = np.zeros(data.shape[1])

    import statsmodels.api as sm

    for vert_idx in range(data.shape[1]):
        X = np.array(behavs)
        X = sm.add_constant(X)
        Y = data[:, vert_idx]
        model = sm.OLS(Y, X)
        results = model.fit()
        coef_a[vert_idx] = results.params[0]
        coef_b[vert_idx] = results.params[1]
        coef_a_t[vert_idx] = results.tvalues[0]
        coef_b_t[vert_idx] = results.tvalues[1]

        X = np.ones(len(behavs))
        Y = data[:, vert_idx]
        model = sm.OLS(Y, X)
        results = model.fit()
        coef_mean[vert_idx] = results.params[0]
        coef_mean_t[vert_idx] = results.tvalues[0]
 

    coef_mean = np.array(coef_mean)
    coef_a = np.array(coef_a)
    coef_b = np.array(coef_b)
    coef_mean_t = np.array(coef_mean_t)
    coef_a_t = np.array(coef_a_t)
    coef_b_t = np.array(coef_b_t)

    coef_mean = coef_mean * weights
    coef_a = coef_a * weights
    coef_b = coef_b * weights
    coef_mean_t = coef_mean_t * weights
    coef_a_t = coef_a_t * weights
    coef_b_t = coef_b_t * weights

    if data.shape[1] > 500:
        vertices = np.array([int(vx) for vx in vertex_list[0]])

    fig = plt.figure()
    plt.rcParams.update({'font.size': 50.0})
    fig.set_size_inches(40, 100)
    fig_dpi = 20

    ax_coef_a = plt.subplot2grid((31, 20), (0, 0), rowspan=5, colspan=18)
    ax_coef_a_cbar = plt.subplot2grid((31, 20), (1, 19), rowspan=3, colspan=1)

    ax_coef_a_t = plt.subplot2grid((31, 20), (6, 0), rowspan=5, colspan=18)
    ax_coef_a_t_cbar = plt.subplot2grid((31, 20), (7, 19), rowspan=3, colspan=1)

    ax_coef_b = plt.subplot2grid((31, 20), (11, 0), rowspan=5, colspan=18)
    ax_coef_b_cbar = plt.subplot2grid((31, 20), (12, 19), rowspan=3, colspan=1)

    ax_coef_b_t = plt.subplot2grid((31, 20), (16, 0), rowspan=5, colspan=18)
    ax_coef_b_t_cbar = plt.subplot2grid((31, 20), (17, 19), rowspan=3, colspan=1)

    ax_coef_mean = plt.subplot2grid((31, 20), (21, 0), rowspan=5, colspan=18)
    ax_coef_mean_cbar = plt.subplot2grid((31, 20), (22, 19), rowspan=3, colspan=1)

    ax_coef_mean_t = plt.subplot2grid((31, 20), (26, 0), rowspan=5, colspan=18)
    ax_coef_mean_t_cbar = plt.subplot2grid((31, 20), (27, 19), rowspan=3, colspan=1)

    if data.shape[1] > 500:
        plot_vol_stc_brainmap(coef_a, vertices, vol_spacing, subjects_dir,
                          ax_coef_a, cap=0.0)
        plot_vol_stc_brainmap(coef_a_t, vertices, vol_spacing, subjects_dir,
                          ax_coef_a_t, cap=0.0)
        plot_vol_stc_brainmap(coef_b, vertices, vol_spacing, subjects_dir,
                          ax_coef_b, cap=0.0)
        plot_vol_stc_brainmap(coef_b_t, vertices, vol_spacing, subjects_dir,
                          ax_coef_b_t, cap=0.0)
        plot_vol_stc_brainmap(coef_mean, vertices, vol_spacing, subjects_dir,
                          ax_coef_mean, cap=0.0)
        plot_vol_stc_brainmap(coef_mean_t, vertices, vol_spacing, subjects_dir,
                          ax_coef_mean_t, cap=0.0)

    else:
        plot_sensor_topomap(coef_a, raw.info, ax_coef_a)
        plot_sensor_topomap(coef_a_t, raw.info, ax_coef_a_t)
        plot_sensor_topomap(coef_b, raw.info, ax_coef_b)
        plot_sensor_topomap(coef_b_t, raw.info, ax_coef_b_t)
        plot_sensor_topomap(coef_mean, raw.info, ax_coef_mean)
        plot_sensor_topomap(coef_mean_t, raw.info, ax_coef_mean_t)


    cmap = mpl.cm.RdBu_r

    norm = MidpointNormalize(np.min(coef_a), np.max(coef_a))
    cb = mpl.colorbar.ColorbarBase(ax_coef_a_cbar, cmap=cmap,
        norm=norm,
        orientation='vertical')
    cb.set_label('Coef a', labelpad=12)

    norm = MidpointNormalize(np.min(coef_a_t), np.max(coef_a_t))
    cb = mpl.colorbar.ColorbarBase(ax_coef_a_t_cbar, cmap=cmap,
        norm=norm,
        orientation='vertical')
    cb.set_label('Coef a t', labelpad=12)

    norm = MidpointNormalize(np.min(coef_b), np.max(coef_b))
    cb = mpl.colorbar.ColorbarBase(ax_coef_b_cbar, cmap=cmap,
        norm=norm,
        orientation='vertical')
    cb.set_label('Coef b', labelpad=12)

    norm = MidpointNormalize(np.min(coef_b_t), np.max(coef_b_t))
    cb = mpl.colorbar.ColorbarBase(ax_coef_b_t_cbar, cmap=cmap,
        norm=norm,
        orientation='vertical')
    cb.set_label('Coef b t', labelpad=12)

    norm = MidpointNormalize(np.min(coef_mean), np.max(coef_mean))
    cb = mpl.colorbar.ColorbarBase(ax_coef_mean_cbar, cmap=cmap,
        norm=norm,
        orientation='vertical')
    cb.set_label('Coef mean', labelpad=12)

    norm = MidpointNormalize(np.min(coef_mean_t), np.max(coef_mean_t))
    cb = mpl.colorbar.ColorbarBase(ax_coef_mean_t_cbar, cmap=cmap,
        norm=norm,
        orientation='vertical')
    cb.set_label('Coef mean t', labelpad=12)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fname = cli_args.identifier + '.png'

        fig.savefig(os.path.join(save_path, fname), dpi=fig_dpi)

