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

from icasso import Icasso

from sklearn.decomposition import PCA
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

    # read contrast data from all participants
    drop_keys = cli_args.drop.split(' ') if cli_args.drop else []

    for fname in sorted(cli_args.coefficients_1):
        if any([key in fname for key in drop_keys]):
            continue

        with open(fname, 'r') as f:
            lines = f.readlines()
            vertex_list.append([val for val in lines[0].strip().split(', ')])
            data_1.append([float(val) for val in lines[1].split(', ')])
            names_1.append(fname.split('/')[-1].split('_')[1])
    data_1 = np.array(data_1)

    for fname in sorted(cli_args.coefficients_2):
        if any([key in fname for key in drop_keys]):
            continue

        with open(fname, 'r') as f:
            lines = f.readlines()
            data_2.append([float(val) for val in lines[1].split(', ')])
            names_2.append(fname.split('/')[-1].split('_')[1])
    data_2 = np.array(data_2)

    for fname in sorted(cli_args.coefficients_norm):
        if any([key in fname for key in drop_keys]):
            continue

        with open(fname, 'r') as f:
            lines = f.readlines()
            norm_data.append([float(val) for val in lines[1].split(', ')])
            names_norm.append(fname.split('/')[-1].split('_')[1])
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
    preica = False
    n_perm = 5000
    random_state = 10

    behav_vars = ['BDI', 'BAI', 'BIS', 'BasTotal']

    n_behav_components = min(len(behav_vars), 3)
    n_contrast_components = 5

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

    if preica:
        print("Whitening with ICA..")

        behav_pca = PCA(n_components=n_behav_components, whiten=True, random_state=random_state)
        behav_wh = behav_pca.fit_transform(np.array(behav_data))
        pca_explained_variance = np.sum(behav_pca.explained_variance_ratio_)

        behav_ica = FastICA(whiten=False, random_state=random_state)
        behav_wh = behav_ica.fit_transform(behav_wh)
        behav_mixing = np.dot(behav_ica.components_, behav_pca.components_)

        exp_var = (np.var(behav_ica.components_, axis=1) / np.sum((np.var(behav_ica.components_, axis=1))) * 
                   pca_explained_variance)
        print("Behav explained variance: " + str(exp_var))
        print("Sum: " + str(np.sum(exp_var)))

        contrast_pca = PCA(n_components=n_contrast_components, whiten=True, random_state=random_state)
        contrast_wh = contrast_pca.fit_transform(np.array(contrast_data))
        pca_explained_variance = np.sum(contrast_pca.explained_variance_ratio_)

        contrast_ica = FastICA(whiten=False, random_state=random_state)
        contrast_wh = contrast_ica.fit_transform(contrast_wh)
        contrast_mixing = np.dot(contrast_ica.components_, contrast_pca.components_)

        exp_var = (np.var(contrast_ica.components_, axis=1) / 
                          np.sum((np.var(contrast_ica.components_, axis=1))) * 
                   pca_explained_variance)
        print("Contrast explained variance: " + str(exp_var))
        print("Sum: " + str(np.sum(exp_var)))

    else:
        print("Whitening with PCA..")
        behav_pca = PCA(n_components=n_behav_components, whiten=True, random_state=random_state)
        behav_wh = behav_pca.fit_transform(np.array(behav_data))
        behav_mixing = behav_pca.components_
        print("Behav explained variance: " + str(behav_pca.explained_variance_ratio_))
        print("Sum: " + str(np.sum(behav_pca.explained_variance_ratio_)))

        contrast_pca = PCA(n_components=n_contrast_components, whiten=True, random_state=random_state)
        contrast_wh = contrast_pca.fit_transform(np.array(contrast_data))
        contrast_mixing = contrast_pca.components_
        print("Contrast explained variance: " + str(contrast_pca.explained_variance_ratio_))
        print("Sum: " + str(np.sum(contrast_pca.explained_variance_ratio_)))

    done = False
    drop_X_component = []
    drop_Y_component = []
    iteration_idx = 0
    while not done:

        X = np.delete(contrast_wh, drop_X_component, axis=1)
        Y = np.delete(behav_wh, drop_Y_component, axis=1)

        n_X_components = X.shape[1]
        n_Y_components = Y.shape[1]

        stacked_data = np.hstack([X, Y])

        print("Decomposing stacked data with PCA..")
        stacked_pca = PCA(n_components=1, random_state=random_state).fit(stacked_data)
        stacked_mixing = stacked_pca.components_

        X_weighted = np.dot(X, stacked_mixing[0, :n_X_components])
        Y_weighted = np.dot(Y, stacked_mixing[0, n_X_components:])

        corrcoef = np.corrcoef(X_weighted,
                               Y_weighted)[0, 1]

        print("Running permutation tests..")
        perm_stats = []
        generator = np.random.RandomState(seed=random_state)
        for perm_idx, ordering in enumerate([generator.permutation(Y.shape[0]) for _ in range(n_perm)]):
            X_test, Y_test = X[ordering, :], Y.copy()
            stacked_data_perm = np.hstack([X_test, Y_test])

            stacked_pca_perm = PCA(n_components=1, random_state=random_state).fit(stacked_data_perm)
            stacked_perm_mixing = stacked_pca_perm.components_

            perm_stat = np.corrcoef(np.dot(X_test, stacked_perm_mixing[0, :n_X_components]),
                                    np.dot(Y_test, stacked_perm_mixing[0, n_X_components:]))[0, 1]
            perm_stats.append(perm_stat)

        pvalue = (len(list(filter(bool, perm_stats > corrcoef)))+1) / (n_perm+1)

        print("Corrcoef with {0} contrast comps and {1} behav comps: {2}, pvalue {3}".format(n_X_components, n_Y_components, corrcoef, pvalue))

        # Plot the stuff

        contrast_weights = np.dot(stacked_mixing[0, :n_X_components], 
                                  np.delete(contrast_mixing, drop_X_component, axis=0))

        behav_weights = np.dot(stacked_mixing[0, n_X_components:], 
                               np.delete(behav_mixing, drop_Y_component, axis=0))

        # convert from mainly blue to mainly red
        if np.mean(contrast_weights) < 0:
            contrast_weights = -contrast_weights
            behav_weights = -behav_weights

        plt.rcParams.update({'font.size': 35.0})

        fig = plt.figure()
        ax_brain = plt.subplot2grid((5, 1), (0, 0), rowspan=1)
        ax_behav = plt.subplot2grid((5, 1), (2, 0), rowspan=1)
        ax_reg = plt.subplot2grid((5, 1), (4, 0), rowspan=1)

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

        ax_reg.scatter(X_weighted, Y_weighted, s=75)

        left = np.min(X_weighted) - np.max(np.abs(X_weighted))*0.1
        right = np.max(X_weighted) + np.max(np.abs(X_weighted))*0.1

        a, b = np.polyfit(X_weighted, Y_weighted, 1)
        ax_reg.plot(np.linspace(left, right, 2), a*np.linspace(left, right, 2) + b)

        ax_reg.set_xlim(left, right)
        ax_reg.set_ylim(np.min(Y_weighted) - np.max(np.abs(Y_weighted))*0.4,
                        np.max(Y_weighted) + np.max(np.abs(Y_weighted))*0.4)

        ax_reg.set_ylabel('Behavioral correlate (AU)')
        ax_reg.yaxis.label.set_size(40)
        ax_reg.set_xlabel('Brain corralate (AU)')
        ax_reg.xaxis.label.set_size(40)
        ax_reg.xaxis.set_tick_params(labelsize=30)
        ax_reg.xaxis.label.set_size(40)
        ax_reg.yaxis.set_tick_params(labelsize=30)
        ax_reg.yaxis.label.set_size(40)

        title = 'CCA iteration {0}, correlation {1}, pvalue {2}'.format(iteration_idx+1, corrcoef, pvalue)
        fig.suptitle(title, fontsize=30.0)

        if save_path:
            path = os.path.join(save_path, 'comps')
            if not os.path.exists(path):
                os.makedirs(path)
            fname = ('cca_' + str(cli_args.identifier) + '_' + 
                     str(iteration_idx+1).zfill(2) + '.png')
            fig.savefig(os.path.join(path, fname))


        X_comp_coefs = []
        for X_idx, X_component in enumerate(contrast_wh.T):
            if X_idx in drop_X_component:
                continue
            comp_coef = np.corrcoef(X_component, Y_weighted)[0, 1]
            X_comp_coefs.append((comp_coef, X_idx))

        Y_comp_coefs = []
        for Y_idx, Y_component in enumerate(behav_wh.T):
            if Y_idx in drop_Y_component:
                continue
            comp_coef = np.corrcoef(Y_component, X_weighted)[0, 1]
            Y_comp_coefs.append((comp_coef, Y_idx))

        X_comp_coefs = sorted(X_comp_coefs, key=lambda x: np.abs(x[0]))
        Y_comp_coefs = sorted(Y_comp_coefs, key=lambda y: np.abs(y[0]))

        if np.abs(X_comp_coefs[0][0]) <= np.abs(Y_comp_coefs[0][0]):
            drop_X_component.append(X_comp_coefs[0][1])
        else:
            drop_Y_component.append(Y_comp_coefs[0][1])

        if len(drop_X_component) == contrast_wh.shape[1] or len(drop_Y_component) == behav_wh.shape[1]:
            done = True

        iteration_idx += 1

