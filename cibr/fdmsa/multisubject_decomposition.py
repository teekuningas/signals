PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    # matplotlib.rc('font', size=15.0)
    matplotlib.use('Agg')

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import matplotlib as mpl
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams.update({'font.size': 10.0})
plt.rcParams.update({'lines.linewidth': 3.0})

import pyface.qt

import sys
import argparse
import os

from collections import OrderedDict

import nibabel as nib
import mne
import numpy as np
import sklearn
import pandas as pd
import scipy.stats

from nilearn.plotting import plot_glass_brain

from sklearn.decomposition import PCA

from scipy.signal import hilbert
from scipy.signal import decimate

from icasso import Icasso

from signals.cibr.common import preprocess
from signals.cibr.common import plot_vol_stc_brainmap

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def plot_barcharts(save_path, components, savename, names):
    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    y_pos = np.arange(len(names))

    plt.rcParams.update({'font.size': 12.0})
    plt.rcParams.update({'lines.linewidth': 3.0})

    fig_, ax = plt.subplots()

    ax.bar(y_pos, components, align='center', alpha=0.5)
    ax.set_xticks(y_pos)
    ax.set_xticklabels(names)
    ax.set_ylabel('PCA weights')

    if not save_path:
        plt.show()

    if save_path:
        weight_path = os.path.join(save_path, 'weights')
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
        name = savename + '.png'

        path = os.path.join(weight_path, name)
        fig_.savefig(path, dpi=155)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path')
    parser.add_argument('--coefficients', nargs='+')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    data = []
    vertex_list = []

    for fname in sorted(cli_args.coefficients):
       with open(fname, 'r') as f:
            lines = f.readlines()
            vertex_list.append([int(val) for val in lines[0].strip().split(', ')])
            data.append([float(val) for val in lines[1].split(', ')])

    data = np.array(data)
    vertex_list = np.array(vertex_list)

    names = []
    for fname in cli_args.coefficients:
        if fname.split('/')[-1].split('_')[1].startswith('V'):
            names.append(fname.split('/')[-1].split('_')[1])
        else:
            names.append('_'.join(fname.split('/')[-1].split('_')[1:3]))

    vertices = vertex_list[0]

    n_brain_components = 3

    ## AVERAGES

    average_brainmap = np.mean(data, axis=0)

    plot_vol_stc_brainmap(save_path, 'average_brainmap', 
        average_brainmap, vertices, '10', subjects_dir)

    ## PCA

    # figure out explained variance
    brain_pca = PCA(n_components=n_brain_components, whiten=False)
    brain_pca_comps = brain_pca.fit_transform(data) 
    orig_total_variance = np.sum(np.diag(np.cov(data.T)))
    try:
        new_vars = np.diag(np.cov(brain_pca_comps.T))
    except:
        new_vars = np.var(brain_pca_comps)
    print("Explained pca brain var: " + str(new_vars/orig_total_variance))
    print("Sum: " + str(np.sum(new_vars)/orig_total_variance))
    pca_explained_variance = np.sum(new_vars) / orig_total_variance

    # do something.. probably ensuring that component variances are 1
    factors = np.std(brain_pca_comps, axis=0)
    brain_pca_comps = np.divide(brain_pca_comps, factors)
    brain_pca_mixing = np.multiply(np.linalg.pinv(brain_pca.components_), factors)
    brain_pca_unmixing = np.divide(brain_pca.components_.T, factors).T

    # convert from mainly blue to mainly red always
    for idx in range(n_brain_components):
        if np.mean(brain_pca_mixing[:, idx]) < 0:
            brain_pca_mixing[:, idx] = -brain_pca_mixing[:, idx]
            brain_pca_unmixing[idx, :] = -brain_pca_unmixing[idx, :]
            brain_pca_comps[:, idx] = -brain_pca_comps[:, idx]

    baselines = []
    for idx in range(n_brain_components):
        baseline = np.zeros(brain_pca_mixing.shape[0])
        baseline = baseline - brain_pca.mean_
        baseline = np.dot(brain_pca_unmixing[idx, :], baseline)
        baselines.append(baseline)

    plt.rcParams.update({'font.size': 13.0})
    plt.rcParams.update({'lines.linewidth': 3.0})

    y = [np.random.normal() for idx in range(brain_pca_comps.shape[0])] 
    for idx in range(n_brain_components):
        fig, ax = plt.subplots()
        fig.suptitle("PCA component " + str(idx+1))
        x = brain_pca_comps[:, idx]

        ax.scatter(x, y, 
                   color='b', s=13)
        ax.axvline(baselines[idx], linewidth=2, color='r')
        ax.set_yticklabels([])

        y_distance = np.max(y) - np.min(y)
        ax.set_ylim([np.min(y) - 3.0*y_distance, 
                     np.max(y) + 3.0*y_distance])

        ax.set_xlim([-3.5, 3.5])
        ax.set_aspect(0.03)

        if save_path:
            scat_path = os.path.join(save_path, 'distributions')
            if not os.path.exists(scat_path):
                os.makedirs(scat_path)
            fig.savefig(os.path.join(
                scat_path, 'brain_pca_' + str(idx+1).zfill(2) + '.png'), dpi=160)

    # plot mixing matrices overlayed to brain
    for idx in range(brain_pca_comps.shape[1]):
        fig, ax = plt.subplots()
        plot_vol_stc_brainmap(None, 'brain_pca_' + str(idx+1).zfill(2), 
                              brain_pca_mixing[:, idx], vertices, '10', subjects_dir,
                              axes=ax)

        fig.suptitle('PCA component ' + str(idx+1))
        if save_path:
            brain_path = os.path.join(save_path, 'vol_brains')
            if not os.path.exists(brain_path):
                os.makedirs(brain_path)
            path = os.path.join(brain_path, 'brain_pca_' + str(idx+1).zfill(2) + '.png')
            fig.savefig(path, dpi=310)

    # t-test components
    for idx in range(n_brain_components):
        title = 'brain ' + str(idx+1) + ' t-test: '
        result = scipy.stats.ttest_1samp(brain_pca_comps[:, idx] - baselines[idx], 0)
        print(title + str(result))

    plt.rcParams.update({'font.size': 13.0})
    plt.rcParams.update({'lines.linewidth': 3.0})


    # plot depressed against healthy
    for comp_idx in range(brain_pca_comps.shape[1]):
        pd_comps_data = []
        for sub_idx in range(brain_pca_comps.shape[0]):
            if names[sub_idx].startswith('V'):
                depressed = False
            else:
                depressed = True

            pd_comps_data.append(
                [names[sub_idx], 
                 brain_pca_comps[sub_idx, comp_idx],
                 depressed])

        pd_comps = pd.DataFrame(pd_comps_data,
                                columns=['name', 'component_score',
                                         'depressed'])

        fig, ax = plt.subplots()
        fig.suptitle('PCA component ' + str(comp_idx+1))
        sns.boxplot(x='depressed', y='component_score', data=pd_comps,
                    ax=ax, whis=3)
        sns.swarmplot(x='depressed', y='component_score', data=pd_comps, color=".25", ax=ax)

        if save_path:
            box_path = os.path.join(save_path, 'boxplots')
            if not os.path.exists(box_path):
                os.makedirs(box_path)
            fig.savefig(os.path.join(
                box_path, 'brain_pca_' + str(comp_idx+1).zfill(2) + '.png'), dpi=160)

    ## ICA

    from sklearn.decomposition import FastICA
    brain_ica = FastICA(n_components=n_brain_components, max_iter=10000)
    brain_ica_comps = brain_ica.fit_transform(data)

    factors = np.std(brain_ica_comps, axis=0)
    brain_ica_comps = np.divide(brain_ica_comps, factors)
    brain_ica_mixing = np.multiply(np.linalg.pinv(brain_ica.components_), factors)
    brain_ica_unmixing = np.divide(brain_ica.components_.T, factors).T

    ica_vars = np.var(brain_ica_mixing, axis=0)
    ica_vars = ica_vars / np.sum(ica_vars)
    ica_vars = ica_vars * pca_explained_variance
    print("ICA explained vars: " + str(ica_vars))

    # convert from mainly blue to mainly red always
    for idx in range(n_brain_components):
        if np.mean(brain_ica_mixing[:, idx]) < 0:
            brain_ica_mixing[:, idx] = -brain_ica_mixing[:, idx]
            brain_ica_unmixing[idx, :] = -brain_ica_unmixing[idx, :]
            brain_ica_comps[:, idx] = -brain_ica_comps[:, idx]

    baselines = []
    for idx in range(n_brain_components):
        baseline = np.zeros(brain_ica_mixing.shape[0])
        baseline = baseline - brain_ica.mean_
        baseline = np.dot(brain_ica_unmixing[idx, :], baseline)
        baselines.append(baseline)

    plt.rcParams.update({'font.size': 13.0})
    plt.rcParams.update({'lines.linewidth': 3.0})

    y = [np.random.normal() for idx in range(brain_ica_comps.shape[0])] 
    for idx in range(n_brain_components):
        fig, ax = plt.subplots()
        fig.suptitle("ICA component " + str(idx+1))
        x = brain_ica_comps[:, idx]

        ax.scatter(x, y, 
                   color='b', s=13)
        ax.axvline(baselines[idx], linewidth=2, color='r')
        ax.set_yticklabels([])

        y_distance = np.max(y) - np.min(y)
        ax.set_ylim([np.min(y) - 3.0*y_distance, 
                     np.max(y) + 3.0*y_distance])

        ax.set_xlim([-3.5, 3.5])
        ax.set_aspect(0.03)

        if save_path:
            scat_path = os.path.join(save_path, 'distributions')
            if not os.path.exists(scat_path):
                os.makedirs(scat_path)
            fig.savefig(os.path.join(
                scat_path, 'brain_ica_' + str(idx+1).zfill(2) + '.png'), dpi=160)

    for idx in range(n_brain_components):
        fig, ax = plt.subplots()
        plot_vol_stc_brainmap(None, 'brain_ica_' + str(idx+1).zfill(1), 
                              brain_ica_mixing[:, idx], vertices, '10', subjects_dir,
                              axes=ax)
        fig.suptitle('ICA component ' + str(idx+1))
        if save_path:
            brain_path = os.path.join(save_path, 'vol_brains')
            if not os.path.exists(brain_path):
                os.makedirs(brain_path)
            path = os.path.join(brain_path, 'brain_ica_' + str(idx+1).zfill(2) + '.png')
            fig.savefig(path, dpi=310)

    for idx in range(n_brain_components):
        title = 'brain ' + str(idx+1) + ' t-test: '
        result = scipy.stats.ttest_1samp(brain_ica_comps[:, idx] - baselines[idx], 0)
        print(title + str(result))

    # plot depressed against healthy
    for comp_idx in range(brain_ica_comps.shape[1]):

        pd_comps_data = []
        for sub_idx in range(brain_ica_comps.shape[0]):
            if names[sub_idx].startswith('V'):
                depressed = False
            else:
                depressed = True

            pd_comps_data.append(
                [names[sub_idx], 
                 brain_ica_comps[sub_idx, comp_idx],
                 depressed])
        pd_comps = pd.DataFrame(pd_comps_data,
                                columns=['name', 'component_score',
                                         'depressed'])

        fig, ax = plt.subplots()
        fig.suptitle('ICA component ' + str(comp_idx+1))
        sns.boxplot(x='depressed', y='component_score', data=pd_comps,
                    ax=ax, whis=3)
        sns.swarmplot(x='depressed', y='component_score', data=pd_comps, color=".25", ax=ax)

        if save_path:
            box_path = os.path.join(save_path, 'boxplots')
            if not os.path.exists(box_path):
                os.makedirs(box_path)
            fig.savefig(os.path.join(
                box_path, 'brain_ica_' + str(comp_idx+1).zfill(2) + '.png'), dpi=160)


    raise Exception('Kissa')

