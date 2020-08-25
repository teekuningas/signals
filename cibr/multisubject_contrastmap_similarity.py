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
    parser.add_argument('--coefficients_alpha_1', nargs='+')
    parser.add_argument('--coefficients_alpha_2', nargs='+')
    parser.add_argument('--coefficients_alpha_norm', nargs='+')
    parser.add_argument('--coefficients_beta_1', nargs='+')
    parser.add_argument('--coefficients_beta_2', nargs='+')
    parser.add_argument('--coefficients_beta_norm', nargs='+')



    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    vertex_list = []

    # meditaatio
    def name_from_fname(fname):
        return fname.split('/')[-1].split('_')[1]
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

    def read_data(coefficients_path):
        data = []
        names = []

        for fname in sorted(coefficients_path):
            if any([key in fname for key in drop_keys]):
                continue

            with open(fname, 'r') as f:
                lines = f.readlines()
                vertex_list.append([val for val in lines[0].strip().split(', ')])
                data.append([float(val) for val in lines[1].split(', ')])
                names.append(name_from_fname(fname))

        data = np.array(data)
        return data, names

    data_alpha_1, names_alpha_1 = read_data(cli_args.coefficients_alpha_1)
    data_alpha_2, names_alpha_2 = read_data(cli_args.coefficients_alpha_2)
    data_alpha_norm, names_alpha_norm = read_data(cli_args.coefficients_alpha_norm)
    data_beta_1, names_beta_1 = read_data(cli_args.coefficients_beta_1)
    data_beta_2, names_beta_2 = read_data(cli_args.coefficients_beta_2)
    data_beta_norm, names_beta_norm = read_data(cli_args.coefficients_beta_norm)

    if not (names_alpha_1 == names_alpha_2 == names_alpha_norm == names_beta_1 == names_beta_2 == names_beta_norm):
        raise Exception('Names do not match')

    names = names_alpha_1

    contrast_data_alpha = np.mean([data_alpha_1, data_alpha_2], axis=0)
    contrast_data_beta = np.mean([data_beta_1, data_beta_2], axis=0)

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
    n_contrast_components = 5
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
        weights_alpha = np.abs(np.mean(data_alpha_norm, axis=0))
        weights_alpha = weights_alpha / np.max(weights_alpha)
        contrast_data_alpha = np.array([scipy.stats.rankdata(elem) for elem in contrast_data_alpha.T]).T * weights_alpha

        weights_beta = np.abs(np.mean(data_beta_norm, axis=0))
        weights_beta = weights_beta / np.max(weights_beta)
        contrast_data_beta = np.array([scipy.stats.rankdata(elem) for elem in contrast_data_beta.T]).T * weights_beta

        behav_data = np.array([scipy.stats.rankdata(elem) for elem in behav_data.T]).T

    contrast_data = np.concatenate([contrast_data_alpha, contrast_data_beta], axis=1)

    contrast_pca = PCA(n_components=n_contrast_components, whiten=True, random_state=random_state)
    contrast_wh = contrast_pca.fit_transform(np.array(contrast_data))
    contrast_mixing = contrast_pca.components_

    # contrast_alpha_pca = PCA(n_components=n_contrast_components, whiten=True, random_state=random_state)
    # contrast_alpha_wh = contrast_alpha_pca.fit_transform(np.array(contrast_data_alpha))
    # contrast_alpha_mixing = contrast_alpha_pca.components_

    # contrast_beta_pca = PCA(n_components=n_contrast_components, whiten=True, random_state=random_state)
    # contrast_beta_wh = contrast_beta_pca.fit_transform(np.array(contrast_data_beta))
    # contrast_beta_mixing = contrast_beta_pca.components_

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

        contrast_weights_alpha = contrast_weights[:int(len(contrast_weights)/2)]
        contrast_weights_beta = contrast_weights[int(len(contrast_weights)/2):]

        # convert from mainly blue to mainly red
        if np.mean(contrast_weights_alpha) < 0:
            contrast_weights_alpha = -contrast_weights_alpha
            contrast_weights_beta = -contrast_weights_beta
            behav_weights = -behav_weights

        print("Correlation coefficient for component " + str(comp_idx+1).zfill(2) + 
              ": " + str(corrcoef))

        print("Behav weights for component " + str(comp_idx+1).zfill(2) + 
              ": " + str(cca_behav_weights))

        print("Contrast weights for component " + str(comp_idx+1).zfill(2) + 
              ": " + str(cca_contrast_weights))

        plt.rcParams.update({'font.size': 35.0})

        fig = plt.figure()
        ax_brain_alpha = plt.subplot2grid((7, 1), (0, 0), rowspan=1)
        ax_brain_beta = plt.subplot2grid((7, 1), (2, 0), rowspan=1)
        ax_behav = plt.subplot2grid((7, 1), (4, 0), rowspan=1)
        ax_reg = plt.subplot2grid((7, 1), (6, 0), rowspan=1)

        # fig.set_size_inches(20, 35)
        fig.set_size_inches(20, 45)
        fig_dpi = 100

        # plot contrast alpha part
        if len(contrast_weights_alpha) > 500:
            vertices = np.array([int(vx) for vx in vertex_list[0]])
            plot_vol_stc_brainmap(contrast_weights_alpha, vertices, '10', subjects_dir, ax_brain_alpha, cap=0.90)
        else:
            plot_sensor_topomap(contrast_weights_alpha, raw.info, ax_brain_alpha)

        # plot contrast beta part
        if len(contrast_weights_beta) > 500:
            vertices = np.array([int(vx) for vx in vertex_list[0]])
            plot_vol_stc_brainmap(contrast_weights_beta, vertices, '10', subjects_dir, ax_brain_beta, cap=0.90)
        else:
            plot_sensor_topomap(contrast_weights_beta, raw.info, ax_brain_beta)

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

