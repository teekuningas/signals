PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    # matplotlib.rc('font', size=15.0)
    matplotlib.use('Agg')

import scipy
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams.update({'font.size': 10.0})
plt.rcParams.update({'lines.linewidth': 3.0})

import pyface.qt

import sys
import argparse
import os

import nibabel as nib
import mne
import numpy as np
import sklearn
import pandas as pd
import scipy.stats

from sklearn.decomposition import PCA

from nilearn.plotting import plot_glass_brain

from signals.cibr.common import plot_vol_stc_brainmap

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return scipy.ma.masked_array(scipy.interp(value, x, y))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path')
    parser.add_argument('--questionnaire')
    parser.add_argument('--behav_measure')
    parser.add_argument('--head')
    parser.add_argument('--coefficients', nargs='+')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    data = []
    vertex_list = []

    # read contrast data from all participants
    for fname in sorted(cli_args.coefficients):
       with open(fname, 'r') as f:
            lines = f.readlines()
            vertex_list.append([int(val) for val in lines[0].strip().split(', ')])
            data.append([float(val) for val in lines[1].split(', ')])

    data = np.array(data)
    vertex_list = np.array(vertex_list)

    # find out participant names
    names = []
    for fname in cli_args.coefficients:
        names.append(fname.split('/')[-1].split('_')[1])

    vertices = vertex_list[0]

    # read head positions
    head_positions = []
    with open(cli_args.head, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split(', ')
        for line in lines[1:]:
            elems = [elem.strip('\n') for elem in line.split(', ')]
            head_positions.append(
                [elems[0].split('_')[1].zfill(3)] + 
                elems[1:])

    head_positions = pd.DataFrame(head_positions, columns=header)

    positions = []
    for name in names:
        positions.append(
            float(head_positions[head_positions['name'] == name]['position'].values[0]))

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

    behav_measure = cli_args.behav_measure

    if behav_measure != 'PCA':
        behavs = []
        for name in names:
            behavs.append(
                float(questionnaire[questionnaire['id'] == name][behav_measure].values[0]))
    else:
        behav_data = []
        for name in names:
            df = questionnaire[questionnaire['id'] == name].iloc[:, 1:7]
            behav_data.append(df.values.astype(np.float)[0])
        behav_pca = PCA(n_components=3, whiten=False)
        behav_comps = behav_pca.fit_transform(np.array(behav_data))
        behavs = behav_comps[:, 0]

    ## VOXELWISE CORRELATION

    corrcoefs = []
    for voxel_idx in range(data.shape[1]):
        Y = data[:, voxel_idx]
        X = behavs
        corrcoefs.append(np.corrcoef(X, Y)[0, 1])

    # add colorbar and print to axis
    fig, ax_brain = plt.subplots()

    plot_vol_stc_brainmap(None, 'voxelwise_corrcoefs', 
                          np.array(corrcoefs), vertices, '10', subjects_dir,
                          axes=ax_brain, cap=0.75)

    divider = make_axes_locatable(ax_brain)
    ax_cbar = divider.append_axes("right", size="2%", pad=0.0)

    cmap = mpl.cm.RdBu_r
    norm = MidpointNormalize(np.min(corrcoefs), np.max(corrcoefs))
    cb = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap,
        norm=norm,
        orientation='vertical')

    if save_path:
        brain_path = os.path.join(save_path, 'voxelwise')
        if not os.path.exists(brain_path):
            os.makedirs(brain_path)
        fig.savefig(os.path.join(brain_path, 'voxelwise_corrcoefs.png'), dpi=310)

    ## PCA VARMISTA NYT VIELÄ ETTÄ DATA SYÖTETÄÄN OIKEIN PÄIN PCA:lle
    import pdb; pdb.set_trace()

    print("Computing PCA..")

    n_brain_components = 5

    brain_pca = PCA(n_components=n_brain_components, whiten=False)
    brain_pca_comps = brain_pca.fit_transform(data)
    orig_total_variance = np.sum(np.diag(np.cov(data.T)))
    new_vars = np.diag(np.cov(brain_pca_comps.T))
    pca_explained_variance = new_vars / orig_total_variance
    print("Explained pca brain var: " + str(pca_explained_variance))
    print("Sum: " + str(np.sum(pca_explained_variance)))

    # do validation by comparing pca's without one participant to the 
    # main one with all participants
    for idx in range(data.shape[0]):
        test_data = np.concatenate([data[:idx], data[idx+1:]])
        test_pca = PCA(n_components=n_brain_components, whiten=False)
        test_comps = test_pca.fit_transform(test_data)

        # correlation with main first pca component
        main_mix_first = np.linalg.pinv(brain_pca.components_)[:, 0]
        test_mix = np.linalg.pinv(test_pca.components_)
        coeffs = []
        for ii in range(n_brain_components):
            coeffs.append(np.abs(np.corrcoef(main_mix_first, test_mix[:, ii])[0, 1]))
        coeffs = np.array(coeffs)

        test_total_variance = np.sum(np.diag(np.cov(test_data.T)))
        test_new_vars = np.diag(np.cov(test_comps.T))
        test_explained_variance = test_new_vars / test_total_variance

        print("Results without participant " + str(names[idx]) + ": ")
        print("Explained variance " + str(names[idx]) + ": " + str(test_explained_variance))
        print("Sorted correlations: " + str(-np.sort(-coeffs)))
        print("Correlation order: " + str(np.argsort(-coeffs)))

    # scree plot
    fig, ax = plt.subplots()
    ax.plot(range(1, len(new_vars) + 1, 1), pca_explained_variance)
    ax.set_xlabel('Component')
    ax.set_ylabel('Explained variance')
    if save_path:
        stats_path = os.path.join(save_path, 'stats')
        if not os.path.exists(stats_path):
            os.makedirs(stats_path)
        fig.savefig(os.path.join(stats_path, 'scree_plot.png'))

    # continue analysis, start by ensuring that component variances are 1
    factors =  np.std(brain_pca_comps, axis=0)
    brain_pca_comps = np.divide(brain_pca_comps, factors)
    brain_pca_mixing = np.multiply(np.linalg.pinv(brain_pca.components_), factors)
    brain_pca_unmixing = np.divide(brain_pca.components_.T, factors).T

    # convert from mainly blue to mainly red always
    for idx in range(n_brain_components):
        if np.mean(brain_pca_mixing[:, idx]) < 0:
            brain_pca_mixing[:, idx] = -brain_pca_mixing[:, idx]
            brain_pca_unmixing[idx, :] = -brain_pca_unmixing[idx, :]
            brain_pca_comps[:, idx] = -brain_pca_comps[:, idx]

    # compute baselines (where the zero contrast would end)
    baselines = []
    for idx in range(n_brain_components):
        baseline = np.zeros(brain_pca_mixing.shape[0])
        baseline = baseline - brain_pca.mean_
        baseline = np.dot(brain_pca_unmixing[idx, :], baseline)
        baselines.append(baseline)

    # plot mixing matrices overlayed to brain
    for idx in range(brain_pca_mixing.shape[1]):
        fig, ax = plt.subplots()
        plot_vol_stc_brainmap(None, '',
                              brain_pca_mixing[:, idx], vertices, '10', subjects_dir,
                              axes=ax)
        fig.suptitle('PCA component ' + str(idx+1))
        if save_path:
            brain_path = os.path.join(save_path, 'brainmaps')
            if not os.path.exists(brain_path):
                os.makedirs(brain_path)
            path = os.path.join(brain_path, 'brain_pca_' + str(idx+1).zfill(2) + '.png')
            fig.savefig(path, dpi=310)

    # now the regression for the first component
    X1 = brain_pca_comps[:, 0]
    X2 = positions
    Y = behavs

    # X = np.array([X1, X2]).T
    X = np.array(X1)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()

    print(results.summary())

    ## ICA

    print("Computing ICA..")

    n_brain_components = 5

    from sklearn.decomposition import FastICA
    brain_ica = FastICA(n_components=n_brain_components)
    brain_ica_comps = brain_ica.fit_transform(data)
    brain_ica_mixing = np.linalg.pinv(brain_ica.components_)

    # plot mixing matrices overlayed to brain
    for idx in range(brain_ica_mixing.shape[1]):
        fig, ax = plt.subplots()
        plot_vol_stc_brainmap(None, '',
                              brain_ica_mixing[:, idx], vertices, '10', subjects_dir,
                              axes=ax)
        fig.suptitle('ICA component ' + str(idx+1))
        if save_path:
            brain_path = os.path.join(save_path, 'brainmaps')
            if not os.path.exists(brain_path):
                os.makedirs(brain_path)
            path = os.path.join(brain_path, 'brain_ica_' + str(idx+1).zfill(2) + '.png')
            fig.savefig(path, dpi=310)

    for idx in range(brain_ica_comps.shape[1]):
        print("Regression for ICA component " + str(idx+1) + ": ")
        Y = behavs
        X = np.array(brain_ica_comps[:, idx])
        model = sm.OLS(Y, sm.add_constant(X))
        results = model.fit()
        print(results.summary())

    ## SPATIAL ICA
    
    print("Computing Spatial ICA..")

    n_brain_components = 5

    from sklearn.decomposition import FastICA
    brain_ica = FastICA(n_components=n_brain_components)
    brain_ica_comps = brain_ica.fit_transform(data.T)
    brain_ica_mixing = np.linalg.pinv(brain_ica.components_)

    # plot mixing matrices overlayed to brain
    for idx in range(brain_ica_comps.shape[1]):
        fig, ax = plt.subplots()
        plot_vol_stc_brainmap(None, '',
                              brain_ica_comps[:, idx], vertices, '10', subjects_dir,
                              axes=ax)
        fig.suptitle('ICA component ' + str(idx+1))
        if save_path:
            brain_path = os.path.join(save_path, 'brainmaps')
            if not os.path.exists(brain_path):
                os.makedirs(brain_path)
            path = os.path.join(brain_path, 'brain_spatialica_' + str(idx+1).zfill(2) + '.png')
            fig.savefig(path, dpi=310)

    for idx in range(brain_ica_mixing.shape[1]):
        print("Regression for Spatial ICA component " + str(idx+1) + ": ")
        Y = behavs
        X = np.array(brain_ica_mixing[:, idx])
        model = sm.OLS(Y, sm.add_constant(X))
        results = model.fit()
        print(results.summary())

    raise Exception('MIAU')

