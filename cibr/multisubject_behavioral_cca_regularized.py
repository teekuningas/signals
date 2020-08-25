PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('figure', max_open_warning=0)
    matplotlib.use('Agg')

import pyface.qt

import sys
import argparse
import os
import time

import mne
import scipy
import scipy.sparse
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import matplotlib as mpl
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats

from pypma import cca
from icasso import Icasso

from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FastICA

from signals.cibr.lib.stc import plot_vol_stc_brainmap
from signals.cibr.lib.sensor import plot_sensor_topomap
from signals.cibr.lib.utils import MidpointNormalize

from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    parser.add_argument('--example_raw')
    parser.add_argument('--coefficients_1', nargs='+')
    parser.add_argument('--coefficients_2', nargs='+')
    parser.add_argument('--coefficients_norm', nargs='+')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

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
    # behav_vars = ['BAI', 'BIS', 'BasTotal']
    behav_vars = ['BDI', 'BIS', 'BasTotal']
    # behav_vars = ['BDI', 'BAI', 'BIS', 'BasTotal']


    # # fdmsa
    # def name_from_fname(fname):
    #     if 'heart' in fname and 'note' in fname:
    #         return '_'.join(fname.split('/')[-1].split('_')[1:][:-2])
    #     else:
    #         return '_'.join(fname.split('/')[-1].split('_')[1:][:-1])
    # behav_vars = ['BDI']

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

    contrast_data = np.mean([data_1, data_2], axis=0)

    # for plotting
    raw = mne.io.Raw(cli_args.example_raw, preload=True, verbose='warning')
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

    pretransform = True
    n_perm = 1000
    random_state = 10

    n_behav_components = len(behav_vars)
    n_contrast_components = 4
    n_cca_components = 2

    penalty_contrast = 1.0
    penalty_behav = 0.7

    behav_data = []
    for name in names:
        row = []
        for var_name in behav_vars:
            row.append(float(questionnaire[questionnaire['id'] == name][var_name].values[0]))
        behav_data.append(row)

    behav_data = np.array(behav_data)

    if pretransform:
        print("Pretransforming non-normal variables")

        # weights = np.cbrt(np.abs(np.mean(contrast_data, axis=0)))
        # weights = np.sqrt(np.abs(np.mean(contrast_data, axis=0)))
        weights = np.abs(np.mean(norm_data, axis=0))
        weights = weights / np.max(weights)
        contrast_data = np.array([scipy.stats.rankdata(elem) for elem in contrast_data.T]).T * weights
        behav_data = np.array([scipy.stats.rankdata(elem) for elem in behav_data.T]).T

    contrast_pca = PCA(n_components=n_contrast_components, whiten=True, random_state=random_state)
    contrast_wh = contrast_pca.fit_transform(np.array(contrast_data))
    contrast_mixing = contrast_pca.components_

    behav_demean = (behav_data - np.mean(behav_data, axis=0)) / np.std(behav_data, axis=0)

    U, V, D = cca(contrast_wh, behav_demean, penaltyx=penalty_contrast, 
                  penaltyz=penalty_behav, K=n_cca_components, standardize=False)

    # why does the whiten have effect if left side is not penalized?!
    from statsmodels.multivariate.cancorr import CanCorr
    stats_cca = CanCorr(behav_demean, contrast_wh)
    print(stats_cca.corr_test().summary())

    # PERMUTATION TEST

    # sample stat
    cca_contrast_weights = U[:, 0]
    cca_behav_weights = V[:, 0]
    X = np.dot(contrast_wh, cca_contrast_weights)
    Y = np.dot(behav_demean, cca_behav_weights)
    sample_stat = np.corrcoef(X, Y)[0, 1]

    # permuted stats
    print("Running permutation tests")
    perm_stats = []
    generator = np.random.RandomState(seed=random_state)
    for perm_idx, ordering in enumerate([generator.permutation(behav_demean.shape[0]) 
                                         for _ in range(n_perm)]):
        contrast_perm = contrast_wh
        behav_perm = behav_demean[ordering, :]

        U_perm, V_perm, D_perm = cca(contrast_perm, behav_perm, 
                                     penaltyx=penalty_contrast, 
                                     penaltyz=penalty_behav, 
                                     K=n_cca_components, standardize=False)

        cca_contrast_weights = U_perm[:, 0]
        cca_behav_weights = V_perm[:, 0]

        X = np.dot(contrast_perm, cca_contrast_weights)
        Y = np.dot(behav_perm, cca_behav_weights)

        perm_stats.append(np.corrcoef(X, Y)[0, 1])

    pvalue = len(list(filter(bool, perm_stats > sample_stat))) / n_perm
    print("First correlation: " + str(sample_stat))
    print("Pvalue: " + str(pvalue))

    # PLOTTING
    for comp_idx in range(n_cca_components):

        cca_contrast_weights = U[:, comp_idx]
        cca_behav_weights = V[:, comp_idx]
        X = np.dot(contrast_wh, cca_contrast_weights)
        Y = np.dot(behav_demean, cca_behav_weights)
        corrcoef = np.corrcoef(X, Y)[0, 1]

        contrast_weights = np.dot(cca_contrast_weights, contrast_mixing)
        behav_weights = cca_behav_weights

        # convert from mainly blue to mainly red
        if np.mean(contrast_weights) < 0:
            contrast_weights = -contrast_weights
            behav_weights = -behav_weights

        print("Correlation coefficient for component " + str(comp_idx+1).zfill(2) + 
              ": " + str(corrcoef))

        print("Behav weights for component " + str(comp_idx+1).zfill(2) + 
              ": " + str(cca_behav_weights))

        print("Contrast weights for component " + str(comp_idx+1).zfill(2) + 
              ": " + str(cca_contrast_weights))

        plt.rcParams.update({'font.size': 35.0})

        fig = plt.figure()
        ax_brain = plt.subplot2grid((5, 1), (0, 0), rowspan=1)
        ax_behav = plt.subplot2grid((5, 1), (2, 0), rowspan=1)
        ax_reg = plt.subplot2grid((5, 1), (4, 0), rowspan=1)

        # fig.set_size_inches(20, 35)
        fig.set_size_inches(20, 40)
        fig_dpi = 100

        # plot contrast part
        if len(contrast_weights) > 500:
            vertices = np.array([int(vx) for vx in vertex_list[0]])
            plot_vol_stc_brainmap(contrast_weights, vertices, '10', subjects_dir, ax_brain, cap=0.90)
        else:
            plot_sensor_topomap(contrast_weights, raw.info, ax_brain)

        ax_behav.bar(behav_vars, behav_weights, align='center', alpha=0.5)
        ax_behav.set_ylabel('Weight (AU)')
        ax_behav.yaxis.label.set_size(40)
        ax_behav.yaxis.set_tick_params(labelsize=30)
        ax_behav.set_xlabel('Behavioral variable')
        ax_behav.xaxis.label.set_size(40)
        ax_behav.xaxis.set_tick_params(labelsize=30)

        ax_reg.scatter(X, Y, s=75)

        left = np.min(X) - np.max(np.abs(X))*0.1
        right = np.max(X) + np.max(np.abs(X))*0.1

        a, b = np.polyfit(X, Y, 1)
        ax_reg.plot(np.linspace(left, right, 2), a*np.linspace(left, right, 2) + b)

        ax_reg.set_xlim(left, right)
        ax_reg.set_ylim(np.min(Y) - np.max(np.abs(Y))*0.4,
                        np.max(Y) + np.max(np.abs(Y))*0.4)

        ax_reg.set_ylabel('Behavioral correlate (AU)')
        ax_reg.yaxis.label.set_size(40)
        ax_reg.set_xlabel('Brain corralate (AU)')
        ax_reg.xaxis.label.set_size(40)
        ax_reg.xaxis.set_tick_params(labelsize=30)
        ax_reg.xaxis.label.set_size(40)
        ax_reg.yaxis.set_tick_params(labelsize=30)
        ax_reg.yaxis.label.set_size(40)

        title = 'CCA component {0}, {1:.3g}, {2:.3g}'.format(
            str(comp_idx+1).zfill(2),
            corrcoef,
            pvalue)
        fig.suptitle(title, fontsize=50.0)

        if save_path:
            path = os.path.join(save_path, 'comps')
            if not os.path.exists(path):
                os.makedirs(path)
            fname = ('cca_' + str(cli_args.identifier) + '_' + 
                     str(comp_idx+1).zfill(2) + '.png')
            fig.savefig(os.path.join(path, fname))

    print("Done.")

