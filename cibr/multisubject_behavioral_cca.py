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
import scipy.sparse
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import matplotlib as mpl
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats

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

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    data_1 = []
    data_2 = []
    names_1 = []
    names_2 = []
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

    if names_1 != names_2:
        raise Exception('Names do not match')
    names = names_1

    contrast_data = np.mean([data_1, data_2], axis=0)

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

    pretransform = False
    preica = True
    postica = True
    n_perm = 200
    random_state = 15
    n_behav_components = 3
    n_contrast_components = 5
    n_cca_components = 3
    behav_vars = ['BDI', 'BAI', 'BIS', 'BasTotal']

    behav_data = []
    for name in names:
        row = []
        for var_name in behav_vars:
            row.append(float(questionnaire[questionnaire['id'] == name][var_name].values[0]))
        behav_data.append(row)

    behav_data = np.array(behav_data)

    if pretransform:
        print("Pretransforming non-normal variables")
        # # spearman correlation matrix
        # contrast_data = np.array([scipy.stats.rankdata(elem) for elem in contrast_data.T]).T
        # behav_data = np.array([scipy.stats.rankdata(elem) for elem in behav_data.T]).T

        # transform to more normality
        contrast_data = contrast_data / np.mean(np.abs(contrast_data))
        contrast_data = np.log10(1 + np.max(np.abs(contrast_data)) + contrast_data)

    if preica:
        print("Whitening with ICA..")

        behav_pca = PCA(n_components=n_behav_components, whiten=True)
        behav_wh = behav_pca.fit_transform(np.array(behav_data))
        pca_explained_variance = np.sum(behav_pca.explained_variance_ratio_)

        behav_ica = FastICA(whiten=False, random_state=random_state)
        behav_wh = behav_ica.fit_transform(behav_wh)
        behav_mixing = np.dot(behav_ica.components_, behav_pca.components_)

        exp_var = (np.var(behav_ica.components_, axis=1) / np.sum((np.var(behav_ica.components_, axis=1))) * 
                   pca_explained_variance)
        print("Behav explained variance: " + str(exp_var))
        print("Sum: " + str(np.sum(exp_var)))

        contrast_pca = PCA(n_components=n_contrast_components, whiten=True)
        contrast_wh = contrast_pca.fit_transform(np.array(contrast_data))
        pca_explained_variance = np.sum(contrast_pca.explained_variance_ratio_)

        contrast_ica = FastICA(whiten=False, random_state=random_state)
        contrast_wh = contrast_ica.fit_transform(contrast_wh)
        contrast_mixing = np.dot(contrast_ica.components_, contrast_pca.components_)

        exp_var = (np.var(contrast_ica.components_, axis=1) / np.sum((np.var(contrast_ica.components_, axis=1))) * 
                   pca_explained_variance)
        print("Contrast explained variance: " + str(exp_var))
        print("Sum: " + str(np.sum(exp_var)))

    else:
        print("Whitening with PCA..")
        behav_pca = PCA(n_components=n_behav_components, whiten=True)
        behav_wh = behav_pca.fit_transform(np.array(behav_data))
        behav_mixing = behav_pca.components_
        print("Behav explained variance: " + str(behav_pca.explained_variance_ratio_))
        print("Sum: " + str(np.sum(behav_pca.explained_variance_ratio_)))

        contrast_pca = PCA(n_components=n_contrast_components, whiten=True)
        contrast_wh = contrast_pca.fit_transform(np.array(contrast_data))
        contrast_mixing = contrast_pca.components_
        print("Contrast explained variance: " + str(contrast_pca.explained_variance_ratio_))
        print("Sum: " + str(np.sum(contrast_pca.explained_variance_ratio_)))


    # from statsmodels.multivariate.cancorr import CanCorr
    # cancorr = CanCorr(behav_wh, contrast_wh)
    # print(cancorr.corr_test().summary())

    comp_idx = 0

    X, Y = behav_wh, contrast_wh
    stacked_data = np.hstack([X, Y])

    if postica:
        print("Decomposing stacked data with ICA..")
        stacked_pca = PCA(n_components=n_cca_components, whiten=True)
        stacked_wh = stacked_pca.fit_transform(stacked_data)
        stacked_ica = FastICA(whiten=False, random_state=random_state).fit(stacked_wh)
        stacked_mixing = np.dot(stacked_ica.components_, stacked_pca.components_)
    else:
        print("Decomposing stacked data with PCA..")
        stacked_pca = PCA(n_components=n_cca_components).fit(stacked_data)
        stacked_mixing = stacked_pca.components_

    corrcoefs = [np.corrcoef(np.dot(X, stacked_mixing[idx, :n_behav_components]),
                             np.dot(Y, stacked_mixing[idx, n_behav_components:]))[0, 1] for idx in range(stacked_mixing.shape[0])]

    stacked_mixing = stacked_mixing[np.argsort(-np.array(corrcoefs))]

    sample_stat = np.corrcoef(np.dot(X, stacked_mixing[comp_idx, :n_behav_components]),
                              np.dot(Y, stacked_mixing[comp_idx, n_behav_components:]))[0, 1]

    perm_stats = []
    print("Running permutation tests..")
    for perm_idx, ordering in enumerate([np.random.permutation(behav_wh.shape[0]) for _ in range(n_perm)]):
        if perm_idx % 100 == 0 and perm_idx != 0:
            print(str(perm_idx) + '/' + str(n_perm) + " permutations done.")
        X, Y = behav_wh[ordering, :], contrast_wh
        stacked_data_perm = np.hstack([X, Y])

        if postica:
            stacked_pca_perm = PCA(n_components=n_cca_components, whiten=True)
            stacked_wh_perm = stacked_pca_perm.fit_transform(stacked_data_perm)
            stacked_ica_perm = FastICA(whiten=False, random_state=random_state).fit(stacked_wh_perm)
            stacked_perm_mixing = np.dot(stacked_ica_perm.components_, stacked_pca_perm.components_)

        else:
            stacked_pca_perm = PCA(n_components=n_cca_components).fit(stacked_data_perm)
            stacked_perm_mixing = stacked_pca_perm.components_

        perm_stat = np.corrcoef(np.dot(X, stacked_perm_mixing[comp_idx, :n_behav_components]),
                                np.dot(Y, stacked_perm_mixing[comp_idx, n_behav_components:]))[0, 1]
        perm_stats.append(perm_stat)

    pvalue = len(list(filter(bool, perm_stats > sample_stat))) / n_perm
    print("First correlation: " + str(sample_stat))
    print("Pvalue: " + str(pvalue))

    for comp_idx in range(n_cca_components):

        behav_weights = np.dot(stacked_mixing[comp_idx, :n_behav_components], 
                               behav_mixing)

        contrast_weights = np.dot(stacked_mixing[comp_idx, n_behav_components:], 
                                  contrast_mixing)

        X = np.dot(behav_wh, stacked_mixing[comp_idx, :n_behav_components])
        Y = np.dot(contrast_wh, stacked_mixing[comp_idx, n_behav_components:])

        corrcoef = np.corrcoef(X, Y)[0, 1]

        print("Correlation coefficient for component " + str(comp_idx+1).zfill(2) + ": " + str(corrcoef))

        fig, ax = plt.subplots(3)

        # plot contrast part
        if len(contrast_weights) > 500:
            vertices = np.array([int(vx) for vx in vertex_list[0]])
            plot_vol_stc_brainmap(contrast_weights, vertices, '10', subjects_dir, ax[0])
        else:
            plot_sensor_topomap(contrast_weights, raw.info, ax[0])

        ax[1].bar(behav_vars, behav_weights, align='center', alpha=0.5)

        frame = pd.DataFrame(np.transpose([X, Y]),
                             columns=['Brain component', 'Behavioral component'])

        sns.regplot(x='Brain component', y='Behavioral component',
                data=frame, ax=ax[2], scatter_kws={'s': 5}, line_kws={'lw': 1})

        title = ('CCA component ' + str(comp_idx+1).zfill(1) + 
                 ' (pvalue: ' + str(pvalue).zfill(3) + ')')
        fig.suptitle(title)

        if save_path:
            path = os.path.join(save_path, 'cca_comps')
            if not os.path.exists(path):
                os.makedirs(path)
            fname = ('cca_' + str(cli_args.identifier) + '_' + 
                     str(comp_idx+1).zfill(2) + '.png')
            fig.savefig(os.path.join(path, fname))

    print("miau")

