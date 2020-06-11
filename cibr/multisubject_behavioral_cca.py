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
    postica = False
    n_perm = 2000
    random_state = 10

    behav_vars = ['BDI', 'BAI', 'BIS', 'BasTotal']
    n_behav_components = min(len(behav_vars), 3)

    n_contrast_components = 5

    n_cca_components = 3

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

    X, Y = behav_wh, contrast_wh
    stacked_data = np.hstack([X, Y])

    if postica:
        print("Decomposing stacked data with ICA..")
        stacked_pca = PCA(n_components=n_cca_components, whiten=True, random_state=random_state)
        stacked_wh = stacked_pca.fit_transform(stacked_data)
        stacked_ica = FastICA(whiten=False, random_state=random_state).fit(stacked_wh)
        stacked_mixing = np.dot(stacked_ica.components_, stacked_pca.components_)
    else:
        print("Decomposing stacked data with PCA..")
        stacked_pca = PCA(n_components=n_cca_components, random_state=random_state).fit(stacked_data)
        stacked_mixing = stacked_pca.components_

    # TEST WHERE WE KEEP ONE VARIABLE CONSTANT AND CHANGE OTHERS
    for comp_idx in range(stacked_mixing.shape[0]):
        print("Testing comp " + str(comp_idx+1))

        X_test = np.dot(contrast_data - np.mean(contrast_data, axis=0), contrast_mixing.T)
        X_test = (X_test / contrast_pca.singular_values_) * np.sqrt(X_test.shape[0] - 1)
        X_test = np.dot(X_test, stacked_mixing[comp_idx, n_behav_components:])

        for test_var in behav_vars:
            print("Keeping " + str(test_var) + " constant.")

            results = []
            for factor in np.linspace(-1, 3, 100):

                behav_mixing_test = behav_mixing.copy()
                behav_mixing_test[:, behav_vars.index(test_var)] = behav_mixing_test[:, behav_vars.index(test_var)] * factor
                
                Y_test = np.dot(behav_data - np.mean(behav_data, axis=0), behav_mixing_test.T)
                Y_test = (Y_test / behav_pca.singular_values_) * np.sqrt(Y_test.shape[0] - 1)
                Y_test = np.dot(Y_test, stacked_mixing[comp_idx, :n_behav_components])

                results.append(np.corrcoef(X_test, Y_test)[0,1])

            print("Correlation lims for " + str(factor) + ": " + str(np.min(results)) + ", " + str(np.max(results)))

            # so this is cool, but is there a way to interpret this for weights as in the picture in the article?

    ## IMPLEMENT ITERATIVE PROCESS THAT MAKES SETS OF VARIABLES SMALLER TO GET MORE SIGNIFICANT RESULTS.
    # should we drop original variables or the components? we could do both.. if we drop original variables, we could then dynamically set the n_components.
    # lets try first decreasing the amount of components
    # in whitening we can do different transformations...
    for comp_idx in range(stacked_mixing.shape[0]):
        print("Testing comp " + str(comp_idx+1))

        X_test = np.dot(contrast_data - np.mean(contrast_data, axis=0), contrast_mixing.T)
        X_test = (X_test / contrast_pca.singular_values_) * np.sqrt(X_test.shape[0] - 1)
        X_test = np.dot(X_test, stacked_mixing[comp_idx, n_behav_components:])

        for test_var in behav_vars:
            print("Keeping " + str(test_var) + " constant.")

            results = []
            for factor in np.linspace(-1, 3, 100):

                behav_mixing_test = behav_mixing.copy()
                behav_mixing_test[:, behav_vars.index(test_var)] = behav_mixing_test[:, behav_vars.index(test_var)] * factor
                
                Y_test = np.dot(behav_data - np.mean(behav_data, axis=0), behav_mixing_test.T)
                Y_test = (Y_test / behav_pca.singular_values_) * np.sqrt(Y_test.shape[0] - 1)
                Y_test = np.dot(Y_test, stacked_mixing[comp_idx, :n_behav_components])

                results.append(np.corrcoef(X_test, Y_test)[0,1])

            print("Correlation lims for " + str(factor) + ": " + str(np.min(results)) + ", " + str(np.max(results)))

            # so this is cool, but is there a way to interpret this for weights as in the picture in the article?
 
     


    # SIGNIFICANCE TEST TEST
    # this is the whole procedure from original data to correlate line by line
    for comp_idx in range(stacked_mixing.shape[0]):
        print("Testing comp " + str(comp_idx+1))
        for test_var in behav_vars:
            X_test = np.dot(contrast_data - np.mean(contrast_data, axis=0), contrast_mixing.T)
            X_test = (X_test / contrast_pca.singular_values_) * np.sqrt(X_test.shape[0] - 1)
            X_test = np.dot(X_test, stacked_mixing[comp_idx, n_behav_components:])

            Y_test = behav_data[:, behav_vars.index(test_var)]

            sample_stat = np.corrcoef(X_test, Y_test)[0, 1]
            perm_stats = []

            generator = np.random.RandomState(seed=random_state)
            for perm_idx, ordering in enumerate([generator.permutation(behav_wh.shape[0]) for _ in range(n_perm)]):
                Y_test = Y_test[ordering]
                perm_stats.append(np.corrcoef(X_test, Y_test)[0, 1])

            pvalue = len(list(filter(bool, np.abs(perm_stats) > np.abs(sample_stat)))) / n_perm
            print("Stat (" + test_var + "): " + str(sample_stat) + " (pvalue: " +  str(pvalue) + ")")

    corrcoefs = [np.corrcoef(np.dot(X, stacked_mixing[idx, :n_behav_components]),
                             np.dot(Y, stacked_mixing[idx, n_behav_components:]))[0, 1] 
                 for idx in range(stacked_mixing.shape[0])]
    stacked_mixing = stacked_mixing[np.argsort(-np.array(corrcoefs))]

    sample_stat = np.corrcoef(np.dot(X, stacked_mixing[0, :n_behav_components]),
                              np.dot(Y, stacked_mixing[0, n_behav_components:]))[0, 1]

    perm_stats = []
    print("Running permutation tests..")
    generator = np.random.RandomState(seed=random_state)
    for perm_idx, ordering in enumerate([generator.permutation(behav_wh.shape[0]) for _ in range(n_perm)]):
        X, Y = behav_wh[ordering, :], contrast_wh
        stacked_data_perm = np.hstack([X, Y])

        if postica:
            stacked_pca_perm = PCA(n_components=n_cca_components, whiten=True, random_state=random_state)
            stacked_wh_perm = stacked_pca_perm.fit_transform(stacked_data_perm)
            stacked_ica_perm = FastICA(whiten=False, random_state=random_state).fit(stacked_wh_perm)
            stacked_perm_mixing = np.dot(stacked_ica_perm.components_, stacked_pca_perm.components_)

        else:
            stacked_pca_perm = PCA(n_components=n_cca_components, random_state=random_state).fit(stacked_data_perm)
            stacked_perm_mixing = stacked_pca_perm.components_

        perm_stat = np.corrcoef(np.dot(X, stacked_perm_mixing[0, :n_behav_components]),
                                np.dot(Y, stacked_perm_mixing[0, n_behav_components:]))[0, 1]
        perm_stats.append(perm_stat)

    pvalue = len(list(filter(bool, perm_stats > sample_stat))) / n_perm
    print("First correlation: " + str(sample_stat))
    print("Pvalue: " + str(pvalue))

    # ICASSO test
    iterations = 1000
    mds_decimate = 20
    distance = 0.7

    print("Running ICASSO")
    cicasso = Icasso(PCA, ica_params={'n_components': n_cca_components}, iterations=iterations,
                     bootstrap=True, vary_init=False)

    def bootstrap_fun(data, generator):
        idxs = generator.choice(range(data.shape[0]), size=data.shape[0]-2, replace=False)
        return data[idxs, :]

    def unmixing_fun(pca):
        return pca.components_

    cicasso.fit(data=stacked_data, fit_params={}, bootstrap_fun=bootstrap_fun,
                unmixing_fun=unmixing_fun, random_state=random_state)

    centrotypes, scores = cicasso.get_centrotype_unmixing(distance=distance)
    print(str(len(scores)) + " components for distance " + str(distance) + ": " + str(scores))

    # experimental!!
    # stacked_mixing = centrotypes

    if save_path:
        cicasso_path = os.path.join(save_path, 'cicasso')
        if not os.path.exists(cicasso_path):
            os.makedirs(cicasso_path)

        print("Plotting dendroram..")
        plt.rcParams.update({'font.size': 30.0})
        fig = cicasso.plot_dendrogram(show=False)
        fig.set_size_inches(20, 10)

        ax = fig.gca()

        ax.set_xlabel('')
        ax.axhline(distance)
        ax.yaxis.label.set_size(30)
        ax.yaxis.set_tick_params(labelsize=30)

        path = os.path.join(cicasso_path, cli_args.identifier + '_dendrogram.png')
        fig.savefig(path, dpi=50)

        print("Plotting mds..")
        before = time.time()
        plt.rcParams.update({'font.size': 30.0})
        fig = cicasso.plot_mds(show=False, distance=distance, decimate=mds_decimate)
        fig.set_size_inches(20, 10)

        ax = fig.gca()

        ax.set_ylabel('Y (AU)')
        ax.yaxis.set_tick_params(labelsize=30)

        ax.set_xlabel('X (AU)')
        ax.xaxis.set_tick_params(labelsize=30)

        path = os.path.join(cicasso_path, cli_args.identifier + '_mds.png')
        fig.savefig(path, dpi=50)
        print("MDS lasted for " + str(time.time() - before))

    for comp_idx in range(stacked_mixing.shape[0]):

        behav_weights = np.dot(stacked_mixing[comp_idx, :n_behav_components], 
                               behav_mixing)

        contrast_weights = np.dot(stacked_mixing[comp_idx, n_behav_components:], 
                                  contrast_mixing)

        # convert from mainly blue to mainly red
        if np.mean(contrast_weights) < 0:
            contrast_weights = -contrast_weights
            behav_weights = -behav_weights

        X = np.dot(contrast_wh, stacked_mixing[comp_idx, n_behav_components:])
        Y = np.dot(behav_wh, stacked_mixing[comp_idx, :n_behav_components])

        corrcoef = np.corrcoef(X, Y)[0, 1]

        print("Correlation coefficient for component " + str(comp_idx+1).zfill(2) + 
              ": " + str(corrcoef))

        plt.rcParams.update({'font.size': 35.0})
        # fig, (ax_brain, ax_behav, ax_reg) = plt.subplots(3)

        fig = plt.figure()
        ax_brain = plt.subplot2grid((5, 1), (0, 0), rowspan=1)
        ax_behav = plt.subplot2grid((5, 1), (2, 0), rowspan=1)
        ax_reg = plt.subplot2grid((5, 1), (4, 0), rowspan=1)

        # fig.set_size_inches(20, 35)
        fig.set_size_inches(20, 40)
        fig_dpi = 100

        # ax_brain.set_title('Canonical weights for spatial task-contrast data')
        # ax_behav.set_title('Canonical weights for behavioral data')
        # ax_reg.set_title('Scatter plot of the first canonical correlates')

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

        title = 'CCA component {0}'.format(str(comp_idx+1).zfill(2))
        fig.suptitle(title, fontsize=50.0)

        if save_path:
            path = os.path.join(save_path, 'comps')
            if not os.path.exists(path):
                os.makedirs(path)
            fname = ('cca_' + str(cli_args.identifier) + '_' + 
                     str(comp_idx+1).zfill(2) + '.png')
            fig.savefig(os.path.join(path, fname))

    print("Done.")

