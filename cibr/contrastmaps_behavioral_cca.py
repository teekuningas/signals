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

from pprint import pprint

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

from sparsecca import cca_ipls
from icasso import Icasso


from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FastICA

from sklearn.model_selection import KFold

from signals.cibr.lib.stc import plot_vol_stc_brainmap
from signals.cibr.lib.stc import plot_vol_stc_labels
from signals.cibr.lib.sensor import plot_sensor_topomap
from signals.cibr.lib.utils import MidpointNormalize

from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    parser.add_argument('--questionnaire')
    parser.add_argument('--identifier')
    parser.add_argument('--drop')
    parser.add_argument('--example_raw')
    parser.add_argument('--spacing')
    parser.add_argument('--coefficients_alpha_1', nargs='+')
    parser.add_argument('--coefficients_alpha_2', nargs='+')
    parser.add_argument('--coefficients_alpha_norm', nargs='+')
    parser.add_argument('--coefficients_beta_1', nargs='+')
    parser.add_argument('--coefficients_beta_2', nargs='+')
    parser.add_argument('--coefficients_beta_norm', nargs='+')
    parser.add_argument('--coefficients_theta_1', nargs='+')
    parser.add_argument('--coefficients_theta_2', nargs='+')
    parser.add_argument('--coefficients_theta_norm', nargs='+')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    vertex_list = []

    # meditaatio
    def name_from_fname(fname):
        return fname.split('/')[-1].split('_')[1]
    # behav_vars = ['BDI', 'BIS', 'BasTotal']
    # behav_vars = ['BIS']
    # behav_vars = ['BDI', 'BAI', 'BIS', 'BasTotal']
    behav_vars = ['BAI', 'BIS']
    # behav_vars = ['BIS']

    # # fdmsa
    # def name_from_fname(fname):
    #     if 'heart' in fname and 'note' in fname:
    #         return '_'.join(fname.split('/')[-1].split('_')[1:][:-2])
    #     else:
    #         return '_'.join(fname.split('/')[-1].split('_')[1:][:-1])
    # behav_vars = ['BDI']

    vol_spacing = '10'
    if cli_args.spacing is not None:
        vol_spacing = str(cli_args.spacing)

    # read contrast data from all participants
    drop_keys = cli_args.drop.split(' ') if cli_args.drop else []

    def read_data(coefficients_path):

        data = []
        names = []

        if not coefficients_path:
            return data, names

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
    data_theta_1, names_theta_1 = read_data(cli_args.coefficients_theta_1)
    data_theta_2, names_theta_2 = read_data(cli_args.coefficients_theta_2)
    data_theta_norm, names_theta_norm = read_data(cli_args.coefficients_theta_norm)

    if names_alpha_1:
        if not (names_alpha_1 == names_alpha_2 == names_alpha_norm):
            raise Exception('Alpha names do not match')

    if names_beta_1:
        if not (names_beta_1 == names_beta_2 == names_beta_norm):
            raise Exception('Beta names do not match')
        
    if names_theta_1:
        if not (names_theta_1 == names_theta_2 == names_theta_norm):
            raise Exception('Theta names do not match')

    names = names_alpha_1

    contrast_data_alpha = np.mean([data_alpha_1, data_alpha_2], axis=0)
    contrast_data_beta = np.mean([data_beta_1, data_beta_2], axis=0)
    contrast_data_theta = np.mean([data_theta_1, data_theta_2], axis=0)

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
    n_perm = 500
    random_state = 10

    n_behav_components = len(behav_vars)
    n_contrast_components = 4

    n_cca_components = 2

    # use_bands = ['alpha', 'beta']
    use_bands = ['alpha']

    behav_data = []
    for name in names:
        row = []
        for var_name in behav_vars:
            row.append(float(questionnaire[questionnaire['id'] == name][var_name].values[0]))
        behav_data.append(row)

    behav_data = np.array(behav_data)

    untransformed_behav_data = np.copy(behav_data)

    if pretransform:
        print("Pretransforming non-normal variables")
        if names_alpha_1:
            # weights_alpha = np.abs(np.mean(data_alpha_norm, axis=0))
            # weights_alpha = np.ones(np.array(data_alpha_norm).shape[1])
            weights_alpha = np.abs(np.mean(data_alpha_norm, axis=0))
            weights_alpha = weights_alpha / np.max(weights_alpha)
            contrast_data_alpha = np.array([scipy.stats.rankdata(elem) for elem in contrast_data_alpha.T]).T * weights_alpha

        if names_beta_1:
            # weights_beta = np.abs(np.mean(data_beta_norm, axis=0))
            # weights_beta = np.ones(np.array(data_beta_norm).shape[1])
            weights_beta = np.abs(np.mean(data_beta_norm, axis=0))
            weights_beta = weights_beta / np.max(weights_beta)
            contrast_data_beta = np.array([scipy.stats.rankdata(elem) for elem in contrast_data_beta.T]).T * weights_beta

        if names_theta_1:
            # weights_theta = np.abs(np.mean(data_theta_norm, axis=0))
            # weights_theta = np.ones(np.array(data_theta_norm).shape[1])
            weights_theta = np.abs(np.mean(data_theta_norm, axis=0))
            weights_theta = weights_theta / np.max(weights_theta)
            contrast_data_theta = np.array([scipy.stats.rankdata(elem) for elem in contrast_data_theta.T]).T * weights_theta

        behav_data = np.array([scipy.stats.rankdata(elem) for elem in behav_data.T]).T

    behav_wh = (behav_data - np.mean(behav_data, axis=0)) / np.std(behav_data, axis=0)

    contrast_band_data = {}
    contrast_band_wh = {}
    contrast_band_mixing = {}

    if names_alpha_1:
        contrast_alpha_pca = PCA(
            n_components=n_contrast_components, whiten=True, random_state=random_state)
        contrast_alpha_wh = contrast_alpha_pca.fit_transform(np.array(contrast_data_alpha))
        contrast_alpha_mixing = contrast_alpha_pca.components_

        print("Alpha explained variance: {0}".format(contrast_alpha_pca.explained_variance_ratio_))
        
        contrast_band_data['alpha'] = contrast_data_alpha
        contrast_band_wh['alpha'] = contrast_alpha_wh
        contrast_band_mixing['alpha'] = contrast_alpha_mixing

    if names_beta_1:
        contrast_beta_pca = PCA(
            n_components=n_contrast_components, whiten=True, random_state=random_state)
        contrast_beta_wh = contrast_beta_pca.fit_transform(np.array(contrast_data_beta))
        contrast_beta_mixing = contrast_beta_pca.components_

        print("Beta explained variance: {0}".format(contrast_beta_pca.explained_variance_ratio_))

        contrast_band_data['beta'] = contrast_data_beta
        contrast_band_wh['beta'] = contrast_beta_wh
        contrast_band_mixing['beta'] = contrast_beta_mixing

    if names_theta_1:
        contrast_theta_pca = PCA(
            n_components=n_contrast_components, whiten=True, random_state=random_state)
        contrast_theta_wh = contrast_theta_pca.fit_transform(np.array(contrast_data_theta))
        contrast_theta_mixing = contrast_theta_pca.components_

        print("Theta explained variance: {0}".format(contrast_theta_pca.explained_variance_ratio_))

        contrast_band_data['theta'] = contrast_data_theta
        contrast_band_wh['theta'] = contrast_theta_wh
        contrast_band_mixing['theta'] = contrast_theta_mixing

    contrast_wh = np.zeros((len(names), 0))

    for band_name in use_bands:
        contrast_wh = np.concatenate([contrast_wh, contrast_band_wh[band_name]], axis=1)

    # behav_penalty_grid = np.linspace(0, 0.05, 21)
    # behav_penalty_grid = np.array([0.0])
    behav_penalty_grid = np.array([0.025])
    penalty_behav = np.mean(behav_penalty_grid)
    penalty_behav_ratio=1.0

    # contrast_penalty_grid = np.linspace(0, 8, 41)
    # contrast_penalty_grid = np.linspace(0, 8, 11)
    # contrast_penalty_grid = np.linspace(0, 7.5, 301)
    # contrast_penalty_grid = np.array([0])
    contrast_penalty_grid = np.array([1.475])
    penalty_contrast = np.mean(contrast_penalty_grid)
    penalty_contrast_ratio = 0.0

    # compute weights with guess params
    cca_contrast_weights, cca_behav_weights = cca_ipls(
        contrast_wh, behav_wh, 
        alpha_lambda_ratio=penalty_behav_ratio,
        alpha_lambda=penalty_behav, 
        beta_lambda=penalty_contrast, 
        beta_lambda_ratio=penalty_contrast_ratio,
        standardize=False,
        n_pairs=n_cca_components, glm_impl='pyglmnet')
    
    print("Find penalties by cross validation")

    n_states = 1000

    results = {}
    for cv_behav_penalty in behav_penalty_grid:
        for cv_contrast_penalty in contrast_penalty_grid:
            cv_train_coefs = {}
            cv_test_coefs = {}
            for state in range(n_states):

                train_idx, test_idx = list(KFold(n_splits=2, shuffle=True, random_state=state).split(contrast_wh))[0]

                contrast_train = contrast_wh[train_idx]
                behav_train = behav_wh[train_idx]

                contrast_test = contrast_wh[test_idx]
                behav_test = behav_wh[test_idx]

                cca_contrast_weights_train, cca_behav_weights_train = cca_ipls(
                    contrast_train, behav_train, 
                    alpha_lambda_ratio=penalty_behav_ratio,
                    alpha_lambda=cv_behav_penalty, 
                    beta_lambda_ratio=penalty_contrast_ratio,
                    beta_lambda=cv_contrast_penalty, 
                    standardize=False,
                    n_pairs=n_cca_components, glm_impl='pyglmnet')

                prelim_test_coefs = {}
                prelim_train_coefs = {}
                for comp_idx in range(n_cca_components):
                    prelim_test_coefs[comp_idx] = []
                    prelim_train_coefs[comp_idx] = []

                for ii in range(n_cca_components):
                    norms = []
                    for jj in range(n_cca_components):
                        norms.append(np.min([
                            np.linalg.norm(cca_behav_weights_train[:, ii] - cca_behav_weights[:, jj]),
                            np.linalg.norm(cca_behav_weights_train[:, ii] + cca_behav_weights[:, jj])]))

                    train_X = np.dot(contrast_train, cca_contrast_weights_train[:, ii])
                    train_Y = np.dot(behav_train, cca_behav_weights_train[:, ii])
                    train_coef = np.corrcoef(train_X, train_Y)[0, 1]
                    test_X = np.dot(contrast_test, cca_contrast_weights_train[:, ii])
                    test_Y = np.dot(behav_test, cca_behav_weights_train[:, ii])
                    test_coef = np.corrcoef(test_X, test_Y)[0, 1]

                    norm_idx = np.argmin(norms)
                    norm = norms[norm_idx]

                    prelim_test_coefs[norm_idx].append((test_coef, norm))
                    prelim_train_coefs[norm_idx].append((train_coef, norm))

                for ii in range(n_cca_components):

                    if not ii in cv_test_coefs:
                        cv_test_coefs[ii] = []

                    if not ii in cv_train_coefs:
                        cv_train_coefs[ii] = []

                    n_coefs = len(prelim_test_coefs[ii])
                    if n_coefs == 0:
                        continue
                    elif n_coefs == 1:
                        cv_train_coefs[ii].append(prelim_train_coefs[ii][0][0])
                        cv_test_coefs[ii].append(prelim_test_coefs[ii][0][0])
                    else:
                        smallest_norm_idx = np.argmin([val[1] for val in prelim_test_coefs[ii]])
                        cv_train_coefs[ii].append(prelim_train_coefs[ii][smallest_norm_idx][0])
                        cv_test_coefs[ii].append(prelim_test_coefs[ii][smallest_norm_idx][0])

                if not np.argmin(norms) in cv_test_coefs:
                    cv_test_coefs[to_idx] = []

                if not np.argmin(norms) in cv_train_coefs:
                    cv_train_coefs[to_idx] = []

            for idx in range(n_cca_components):
                if idx not in cv_test_coefs:
                    test_coef = 0
                    train_coef = 0
                    n_succ = 0
                    n_nans = 0

                elif len(cv_test_coefs[idx]) < n_states / 4:
                    not_nans_test = [coef for coef in cv_test_coefs[idx] if not np.isnan(coef)]
                    n_succ = len(not_nans_test)
                    n_nans = len(cv_test_coefs[idx]) - n_succ
                    test_coef = 0
                    train_coef = 0
                else:
                    test_not_nans = [coef for coef in cv_test_coefs[idx] if not np.isnan(coef)]
                    train_not_nans = [coef for coef in cv_train_coefs[idx] if not np.isnan(coef)]
                    n_succ = len(test_not_nans)
                    n_nans = len(cv_test_coefs[idx]) - n_succ
                    test_coef = np.mean(test_not_nans)
                    train_coef = np.mean(train_not_nans)

                print('comp {}, {}, {} (test_coef: {:.4f}, train_coef: {:.4f}, behav: {:.4f}, contrast: {:.4f})'.format(
                    idx+1, n_succ, n_nans, test_coef, train_coef, cv_behav_penalty, cv_contrast_penalty))
                # print(str([str(round(val, 2)) for val in not_nans]))

                if not idx in results:
                    results[idx] = []

                results[idx].append((test_coef, cv_behav_penalty, cv_contrast_penalty))

    for result_idx in range(n_cca_components):
        res = np.array(results[result_idx])

        fig, ax = plt.subplots()

        arr = np.zeros((len(behav_penalty_grid), len(contrast_penalty_grid)))
        for ii in range(len(behav_penalty_grid)):
            for jj in range(len(contrast_penalty_grid)):
                coef = res[np.where((res[:, 2] == contrast_penalty_grid[jj]) & 
                                    (res[:, 1] == behav_penalty_grid[ii]))][0][0]
                arr[ii, jj] = coef

        im = ax.imshow(arr, cmap='Reds')
        fig.colorbar(im, ax=ax)

        ax.set_xlabel('Contrast penalty')
        ax.set_ylabel('Behav penalty')
        ax.set_xticks(range(len(contrast_penalty_grid)))
        ax.set_xticklabels([round(val, 4) for val in contrast_penalty_grid])
        ax.set_yticks(range(len(behav_penalty_grid)))
        ax.set_yticklabels([round(val, 4) for val in behav_penalty_grid])

        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

        if save_path:
            path = os.path.join(save_path, 'heatmaps')
            if not os.path.exists(path):
                os.makedirs(path)
            fname = cli_args.identifier + '_' + str(result_idx+1) + '.png'
            fig.savefig(os.path.join(path, fname))

        fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)

        arr = np.zeros((len(contrast_penalty_grid)))
        for jj in range(len(contrast_penalty_grid)):
            coef = res[np.where((res[:, 2] == contrast_penalty_grid[jj]) & 
                                (res[:, 1] == behav_penalty_grid[0]))][0][0]
            arr[jj] = coef

        ax.plot(contrast_penalty_grid, arr)
        ax.set_xlabel('Contrast penalty')
        ax.set_ylabel('Test score')

        if save_path:
            path = os.path.join(save_path, 'cv_curves')
            if not os.path.exists(path):
                os.makedirs(path)
            fname = cli_args.identifier + '_' + str(result_idx+1) + '.png'
            fig.savefig(os.path.join(path, fname), dpi=50)

            fname = cli_args.identifier + '_' + str(result_idx+1) + '.csv'
            with open(os.path.join(path, fname), 'w') as f:
                f.write(','.join([str(elem) for elem in contrast_penalty_grid]) + '\n')
                f.write(','.join([str(elem) for elem in arr]))


    def find_best(arr):
        arr = arr[~np.isnan(arr[:, 0]), :]
        idx = np.argmax(arr[:, 0])
        best_coef = arr[:, 0][idx]
        best_behav_penalty = arr[:, 1][idx]
        best_contrast_penalty = arr[:, 2][idx]
        return best_coef, best_behav_penalty, best_contrast_penalty

    print("Results from CV:")
    for result_idx in range(n_cca_components):
        print("Component " + str(result_idx+1))
        pprint(results[result_idx])

    for result_idx in range(n_cca_components):
        print("Running rest of the analysis with best weights for comp " + str(result_idx+1))

        plt.rcParams.update({'font.size': 12.0})

        best_cv_coef, best_behav_penalty, best_contrast_penalty = find_best(
            np.array(results[result_idx]))

        print("Best behav penalty for " + str(result_idx+1) + ": " + str(best_behav_penalty))
        print("Best contrast penalty for " + str(result_idx+1) + ": " + str(best_contrast_penalty))
        print("Coef: " + str(best_cv_coef))
        penalty_behav = best_behav_penalty
        penalty_contrast = best_contrast_penalty

        # compute these agian with best params
        cca_contrast_weights, cca_behav_weights = cca_ipls(
            contrast_wh, behav_wh, 
            alpha_lambda_ratio=penalty_behav_ratio,
            alpha_lambda=penalty_behav, 
            beta_lambda=penalty_contrast, 
            beta_lambda_ratio=penalty_contrast_ratio,
            standardize=False,
            n_pairs=n_cca_components, glm_impl='pyglmnet')

        # INFO ON COMPONENTS
        for comp_idx in range(n_cca_components):
            X = np.dot(contrast_wh, cca_contrast_weights[:, comp_idx])
            Y = np.dot(behav_wh, cca_behav_weights[:, comp_idx])
            corrcoef = np.corrcoef(X, Y)[0, 1]

            print("Behav weights for component " + str(comp_idx+1) + ": ")
            print(cca_behav_weights[:, comp_idx])
            print("Correlation for component " + str(comp_idx+1) + ": " + str(corrcoef))

        print("Running permutation tests")
        perm_stats = []
        generator = np.random.RandomState(seed=random_state)
        for perm_idx, ordering in enumerate([generator.permutation(behav_wh.shape[0]) 
                                             for _ in range(n_perm)]):
            if perm_idx % 10 == 0 and perm_idx > 0:
                print(str(perm_idx) + " permutations done.")

            contrast_perm = contrast_wh
            behav_perm = behav_wh[ordering, :]

            cca_contrast_weights_perm, cca_behav_weights_perm = cca_ipls(
                contrast_perm, behav_perm, 
                alpha_lambda_ratio=penalty_behav_ratio,
                alpha_lambda=penalty_behav, 
                beta_lambda=penalty_contrast, 
                beta_lambda_ratio=penalty_contrast_ratio,
                standardize=False,
                n_pairs=n_cca_components, glm_impl='pyglmnet')

            corrcoefs = []
            for comp_idx in range(n_cca_components):
                X = np.dot(contrast_perm, cca_contrast_weights_perm[:, comp_idx])
                Y = np.dot(behav_perm, cca_behav_weights_perm[:, comp_idx])
                corrcoefs.append(np.corrcoef(X, Y)[0, 1])
            perm_stats.append(np.max(corrcoefs))

        pvalues = []
        for comp_idx in range(n_cca_components):
            X = np.dot(contrast_wh, cca_contrast_weights[:, comp_idx])
            Y = np.dot(behav_wh, cca_behav_weights[:, comp_idx])
            sample_stat = np.corrcoef(X, Y)[0, 1]

            pvalue = len(list(filter(bool, perm_stats > sample_stat))) / n_perm
            print("Stats for comp " + str(comp_idx+1) + ": " + str(sample_stat) + ", " + str(pvalue))

            pvalues.append(pvalue)

        # Plotting parcellation values
        for comp_idx in range(n_cca_components):
            for band_idx, band_name in enumerate(use_bands):
                fig, ax = plt.subplots()
                fig.set_size_inches(10, 10)
                contrast_weights = np.dot(
                    cca_contrast_weights[band_idx*n_contrast_components:(band_idx+1)*n_contrast_components, comp_idx],
                    contrast_band_mixing[band_name])

                vertices = np.array([int(vx) for vx in vertex_list[0]])
                plot_vol_stc_labels(contrast_weights, vertices, vol_spacing, subjects_dir, ax, n_labels=10)
                if save_path:
                    path = os.path.join(save_path, 'parc')
                    if not os.path.exists(path):
                        os.makedirs(path)

                    fname = (str(cli_args.identifier) + '_' + 
                        str(comp_idx+1).zfill(2) + '_' +
                        str(band_name) + '_' + 
                        str(round(penalty_behav, 4)).replace('.', '') + '_' +
                        str(round(penalty_contrast, 4)).replace('.', '') +
                        '.png')

                    fig.savefig(os.path.join(path, fname), dpi=100)
 

        print("Plotting CCA components")
        for comp_idx in range(n_cca_components):

            X = np.dot(contrast_wh, cca_contrast_weights[:, comp_idx])
            Y = np.dot(behav_wh, cca_behav_weights[:, comp_idx])
            corrcoef = np.corrcoef(X, Y)[0, 1]

            behav_weights = cca_behav_weights[:, comp_idx]

            contrast_weights = {}
            for band_idx, band_name in enumerate(use_bands):
                contrast_weights[band_name] = np.dot(
                    cca_contrast_weights[band_idx*n_contrast_components:(band_idx+1)*n_contrast_components, comp_idx],
                    contrast_band_mixing[band_name])

            # convert from mainly red to mainly blue
            print("Mean for alpha: {0}".format(np.mean(contrast_weights['alpha'])))
            if np.mean(contrast_weights['alpha']) > 0:
                for band_name in use_bands:
                    contrast_weights[band_name] = -contrast_weights[band_name]
                behav_weights = -behav_weights

            plt.rcParams.update({'font.size': 60.0})

            fig = plt.figure()

            ax_bands = {}
            for band_idx, band_name in enumerate(use_bands):
                ax_bands[band_name] = plt.subplot2grid((len(use_bands)*5 + 5 + 4, 8), (band_idx*5, 0), rowspan=4, colspan=7)

            ax_cbar = plt.subplot2grid((len(use_bands)*5 + 5 + 4, 8), (0, 7), rowspan=len(use_bands)*5-1, colspan=1)

            ax_behav = plt.subplot2grid((len(use_bands)*5 + 5 + 4, 8), (len(use_bands)*5, 0), rowspan=4, colspan=7)
            ax_reg = plt.subplot2grid((len(use_bands)*5 + 5 + 4, 8), (len(use_bands)*5 + 5, 0), rowspan=4, colspan=7)

            fig.set_size_inches(40, 25 + 20*len(use_bands))
            fig_dpi = 15

            vmax = 0
            vmin = 0
            vmax_abs = 0
            for band_name in use_bands:
                vmax_abs_band = np.max(np.abs(contrast_weights[band_name]))
                vmin_band = np.min(contrast_weights[band_name])
                vmax_band = np.max(contrast_weights[band_name])
                if vmax_abs_band > vmax_abs:
                    vmax_abs = vmax_abs_band
                if vmax_band > vmax:
                    vmax = vmax_band
                if vmin_band < vmin:
                    vmin = vmin_band

            cmap = mpl.cm.RdBu_r
            norm = MidpointNormalize(vmin, vmax)
            cb = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap,
                norm=norm,
                orientation='vertical')
            cb.set_label('Weight (AU)', labelpad=10)

            for band_name in use_bands:
                if len(contrast_weights[band_name]) > 500:
                    vertices = np.array([int(vx) for vx in vertex_list[0]])
                    plot_vol_stc_brainmap(contrast_weights[band_name], vertices, vol_spacing, subjects_dir, ax_bands[band_name], 
                                          cap=0.0, vmax=vmax_abs)
                else:
                    factor = np.max(np.abs(contrast_weights[band_name])) / vmax_abs
                    plot_sensor_topomap(contrast_weights[band_name], raw.info, ax_bands[band_name], factor=factor)

                ax_bands[band_name].set_title(band_name)

            ax_behav.bar(behav_vars, behav_weights, align='center', alpha=1.0, width=0.5)
            ax_behav.axhline(0)
            ax_behav.set_ylabel('Weight (AU)')
            ax_behav.yaxis.label.set_size(80)
            ax_behav.yaxis.set_tick_params(labelsize=60)
            ax_behav.set_xlabel('Behavioral variable')
            ax_behav.xaxis.label.set_size(80)
            ax_behav.xaxis.set_tick_params(labelsize=60)

            ax_reg.scatter(X, Y, s=300)

            left = np.min(X) - np.max(np.abs(X))*0.1
            right = np.max(X) + np.max(np.abs(X))*0.1

            a, b = np.polyfit(X, Y, 1)
            ax_reg.plot(np.linspace(left, right, 2), a*np.linspace(left, right, 2) + b)

            ax_reg.set_xlim(left, right)
            ax_reg.set_ylim(np.min(Y) - np.max(np.abs(Y))*0.4,
                            np.max(Y) + np.max(np.abs(Y))*0.4)

            ax_reg.set_ylabel('Behavioral correlate (AU)')
            ax_reg.yaxis.label.set_size(80)
            ax_reg.set_xlabel('Brain correlate (AU)')
            ax_reg.xaxis.label.set_size(80)
            ax_reg.xaxis.set_tick_params(labelsize=60)
            ax_reg.xaxis.label.set_size(80)
            ax_reg.yaxis.set_tick_params(labelsize=60)
            ax_reg.yaxis.label.set_size(80)

            title = 'CCA component {} (b: {}, c: {}, cv: {}), {:.3g}, {:.3g}'.format(
                str(comp_idx+1).zfill(2),
                round(penalty_behav, 4),
                round(penalty_contrast, 4),
                round(best_cv_coef, 3),
                corrcoef,
                pvalues[comp_idx])
            fig.suptitle(title, fontsize=60.0)

            if save_path:
                path = os.path.join(save_path, 'comps')
                if not os.path.exists(path):
                    os.makedirs(path)
                fname = ('cca_' + str(cli_args.identifier) + '_' + 
                         str(comp_idx+1).zfill(2) + '_' +
                         str(round(penalty_behav, 4)).replace('.', '') + '_' +
                         str(round(penalty_contrast, 4)).replace('.', '') +
                         '.png')
                fig.savefig(os.path.join(path, fname), dpi=fig_dpi)


            # individual contributions in each band
            for band_idx, band in enumerate(use_bands):
                fig = plt.figure()

                n_scatters = len(behav_vars)
                ax_brain = plt.subplot2grid((1 + 3 + 3*n_scatters, 9), (1, 0), rowspan=2, colspan=8)
                ax_cbar = plt.subplot2grid((1 + 3 + 3*n_scatters, 9), (1, 8), rowspan=2, colspan=1)
                ax_scatters = []
                for behav_idx in range(len(behav_vars)):
                    ax_scatter = plt.subplot2grid((1 + 3 + 3*n_scatters, 9), (4 + behav_idx*3, 0), rowspan=2, colspan=8)
                    ax_scatters.append(ax_scatter)

                fig.set_size_inches(40, 20 + 20 * n_scatters)
                fig_dpi = 15
                fig.suptitle(str(band))

                brain_weights = contrast_weights[band].copy()

                for behav_idx, behav_var in enumerate(behav_vars):

                    scatter_Y = behav_wh[:, behav_idx]
                    scatter_X = X.copy()

                    ax_scatter = ax_scatters[behav_idx]

                    ax_scatter.scatter(scatter_X, scatter_Y, c='blue', s=500)

                    m, b = np.polyfit(scatter_X, scatter_Y, 1)
                    ax_scatter.plot(scatter_X, m*scatter_X + b)

                    ax_scatter.set_ylabel(behav_var + ' score (AU)')
                    ax_scatter.yaxis.label.set_size(80)
                    ax_scatter.yaxis.set_tick_params(labelsize=60)
                    ax_scatter.yaxis.label.set_size(80)

                    ax_scatter.set_xlabel('Brain correlate (AU)')
                    ax_scatter.xaxis.label.set_size(80)
                    ax_scatter.xaxis.set_tick_params(labelsize=60)
                    ax_scatter.xaxis.label.set_size(80)

                plot_vol_stc_brainmap(brain_weights, vertices, vol_spacing, 
                                      subjects_dir, ax_brain,
                                      cap=0.0, vmax=vmax_abs)

                cmap = mpl.cm.RdBu_r
                norm = MidpointNormalize(vmin, vmax)
                cb = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap,
                    norm=norm,
                    orientation='vertical')
                cb.set_label('Weight (AU)', labelpad=10)

                if save_path:
                    path = os.path.join(save_path, 'individual_contributions_band')
                    if not os.path.exists(path):
                        os.makedirs(path)
                    fname = ('cca_' + str(cli_args.identifier) + '_' + 
                             str(comp_idx+1).zfill(2) + '_' +
                             str(round(penalty_behav, 4)).replace('.', '') + '_' +
                             str(round(penalty_contrast, 4)).replace('.', '') + '_' + 
                             str(band) + 
                             '.png')
                    fig.savefig(os.path.join(path, fname), dpi=fig_dpi)


            # individual contributions separately
            for behav_idx, behav_var in enumerate(behav_vars):
                for band_idx, band in enumerate(use_bands):

                    fig = plt.figure()

                    ax_scatter = plt.subplot2grid((5, 18), (1, 0), rowspan=3, colspan=7)
                    ax_brain = plt.subplot2grid((5, 18), (1, 8), rowspan=3, colspan=8)
                    ax_cbar = plt.subplot2grid((5, 18), (1, 17), rowspan=3, colspan=1)

                    fig.set_size_inches(80, 20)
                    fig_dpi = 15

                    # scatter_Y = untransformed_behav_data[:, behav_idx]
                    scatter_Y = behav_wh[:, behav_idx]
                    scatter_X = X.copy()
                    brain_weights = contrast_weights[band].copy()

                    cbar_vmin = vmin
                    cbar_vmax = vmax

                    behav_weights = behav_weights.copy()
                    if behav_weights[behav_idx] < 0:
                        scatter_X = -scatter_X
                        brain_weights = -brain_weights

                        cbar_vmin = -vmax
                        cbar_vmax = -vmin

                    ax_scatter.scatter(scatter_X, scatter_Y, c='blue', s=500)

                    corrcoef = np.corrcoef(scatter_X, scatter_Y)[0, 1]
                    fig.suptitle('Corrcoef: ' + str(corrcoef))
                    
                    m, b = np.polyfit(scatter_X, scatter_Y, 1)
                    ax_scatter.plot(scatter_X, m*scatter_X + b)

                    ax_scatter.set_ylabel(behav_var + ' score (AU)')
                    # ax_scatter.set_ylabel('Behavioral correlate (AU)')
                    ax_scatter.yaxis.label.set_size(80)
                    ax_scatter.yaxis.set_tick_params(labelsize=60)
                    ax_scatter.yaxis.label.set_size(80)

                    ax_scatter.set_xlabel('Brain correlate (AU)')
                    ax_scatter.xaxis.label.set_size(80)
                    ax_scatter.xaxis.set_tick_params(labelsize=60)
                    ax_scatter.xaxis.label.set_size(80)

                    plot_vol_stc_brainmap(brain_weights, vertices, vol_spacing, 
                                          subjects_dir, ax_brain,
                                          cap=0.0, vmax=vmax_abs)

                    cmap = mpl.cm.RdBu_r
                    norm = MidpointNormalize(cbar_vmin, cbar_vmax)
                    cb = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap,
                        norm=norm,
                        orientation='vertical')
                    cb.set_label('Weight (AU)', labelpad=10)

                    if save_path:
                        path = os.path.join(save_path, 'individual_contributions')
                        if not os.path.exists(path):
                            os.makedirs(path)
                        fname = ('cca_' + str(cli_args.identifier) + '_' + 
                                 str(comp_idx+1).zfill(2) + '_' +
                                 str(round(penalty_behav, 4)).replace('.', '') + '_' +
                                 str(round(penalty_contrast, 4)).replace('.', '') + '_' + 
                                 str(band) + '_' +
                                 str(behav_var) +
                                 '.png')
                        fig.savefig(os.path.join(path, fname), dpi=fig_dpi)

    print("Done.")

