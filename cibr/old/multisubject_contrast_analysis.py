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
    parser.add_argument('--behav')
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

    names = [fname.split('/')[-1].split('_')[1] for fname in cli_args.coefficients]

    # for idx in range(data.shape[0]):
    #     max_ = np.max(np.abs(data[idx]))
    #     data[idx] /= max_

    vertices = vertex_list[0]

    n_brain_components = 3
    n_behav_components = 3

    behav_info = []
    with open(cli_args.behav, 'r') as f:
        lines = f.readlines()
        behav_info_header = ['id'] + [elem.strip('\n') for elem in lines[0][1:].split(',')]
        for line in lines[1:]:
            elems = [elem.strip('\n') for elem in line.split(',')]
            behav_info.append([elems[0].zfill(3)] + elems[1:])

    behav_info = pd.DataFrame(behav_info, columns=behav_info_header)
    behav_info = behav_info.loc[behav_info['id'].isin(names)]
    behav_data = behav_info.iloc[:, 1:7].values.astype(np.float)

    behav_pca = PCA(n_components=n_behav_components, whiten=False)
    behav_comps = behav_pca.fit_transform(behav_data) 
    orig_total_variance = np.sum(np.diag(np.cov(behav_data.T)))
    try:
        new_vars = np.diag(np.cov(behav_comps.T))
    except:
        new_vars = np.var(behav_comps)
    print("Explained behav var: " + str(new_vars/orig_total_variance))
    print("Sum: " + str(np.sum(new_vars/orig_total_variance)))

    behav_pca_wh = PCA(n_components=n_behav_components, whiten=True)
    behav_comps_wh = behav_pca_wh.fit_transform(behav_data) 
    behav_mixing = np.linalg.pinv(behav_pca_wh.components_)

    print("Plotting behav pcasso")
    for idx in range(n_behav_components):
        plot_barcharts(save_path, behav_mixing[:, idx], 
                       'behav_' + str(idx+1).zfill(2),
                       behav_info_header[1:7])

    ## BEHAV PCASSO

    pca_params = {
        'n_components': n_behav_components,
    }

    pcasso = Icasso(PCA, ica_params=pca_params, iterations=behav_data.shape[0],
                    bootstrap=True, vary_init=False)

    counter = 0
    def bootstrap_fun(data, generator):
        global counter
        indices = [val for val in range(data.shape[0])]; del indices[counter]
        counter += 1
        return data[indices, :]

    def unmixing_fun(pca):
        return pca.components_

    fit_params = {}

    pcasso.fit(data=behav_data, fit_params=fit_params, bootstrap_fun=bootstrap_fun,
               unmixing_fun=unmixing_fun)

    if save_path:
        pcasso_path = os.path.join(save_path, 'pcasso')
        if not os.path.exists(pcasso_path):
            os.makedirs(pcasso_path)

    plt.rcParams.update({'font.size': 30.0})
    plt.rcParams.update({'lines.linewidth': 4.0})

    print("Plotting dendrogram")
    if save_path:
        fig = pcasso.plot_dendrogram(show=False)
        path = os.path.join(pcasso_path, 'behav_dendrogram.png')
        fig.gca().set_xlabel('')
        fig.savefig(path, dpi=60)
    else:
        pcasso.plot_dendrogram()

    print("Plotting mds")
    if save_path:
        fig = pcasso.plot_mds(show=False)
        path = os.path.join(pcasso_path, 'behav_mds.png')
        fig.savefig(path, dpi=60)
    else:
        pcasso.plot_mds(show=False)

    behav_pcasso_unmixing, behav_pcasso_scores = (
        pcasso.get_centrotype_unmixing(distance=0.7))

    print("Behav pcasso scores")
    print(str(behav_pcasso_scores))

    print("Plotting behav pcasso")
    for idx in range(n_behav_components):
        plot_barcharts(save_path, behav_pcasso_unmixing[idx, :], 
                       'behav_pcasso_' + str(idx+1).zfill(2),
                       behav_info_header[1:7])

    ## AVERAGES

    average_brainmap = np.mean(data, axis=0)

    plot_vol_stc_brainmap(save_path, 'average_brainmap', 
        average_brainmap, vertices, '10', subjects_dir)

    bound = np.max([np.abs(np.min(average_brainmap)), 
                    np.abs(np.max(average_brainmap))])

    plt.rcParams.update({'font.size': 15.0})
    plt.rcParams.update({'lines.linewidth': 3.0})

    if save_path:
        colorbar_path = os.path.join(save_path, 'colorbars')
        if not os.path.exists(colorbar_path):
            os.makedirs(colorbar_path)
    fig, ax = plt.subplots()

    cmap = plt.cm.RdBu_r
 
    norm = mpl.colors.Normalize(vmin=-bound,vmax=bound)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    plt.colorbar(mappable, ax=ax, aspect=8)

    if save_path:
        fig.savefig(os.path.join(colorbar_path, 'average.png'), dpi=100)

    ## PCA

    brain_pca = PCA(n_components=n_brain_components, whiten=False)
    brain_pca_comps = brain_pca.fit_transform(data) 
    orig_total_variance = np.sum(np.diag(np.cov(data.T)))
    try:
        new_vars = np.diag(np.cov(brain_pca_comps.T))
    except:
        new_vars = np.var(brain_pca_comps)
    print("Explained pca brain var: " + str(new_vars/orig_total_variance))
    print("Sum: " + str(np.sum(new_vars/orig_total_variance)))
    pca_explained_variance = np.sum(new_vars) / orig_total_variance

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
        x = brain_pca_comps[:, idx]

        ax.scatter(x, y, 
                   color='b', s=13)
        ax.axvline(baselines[idx], linewidth=2, color='r')
        ax.set_yticklabels([])

        y_distance = np.max(y) - np.min(y)
        ax.set_ylim([np.min(y) - 3.0*y_distance, 
                     np.max(y) + 3.0*y_distance])

        # x_distance = np.max(x) - np.min(x)
        # ax.set_xlim([np.min(x) - 0.1*x_distance, 
        #              np.max(x) + 0.1*x_distance])
        # we know that the values should be close to zero so
        ax.set_xlim([-3.5, 3.5])
        ax.set_aspect(0.03)

        if save_path:
            scat_path = os.path.join(save_path, 'distributions')
            if not os.path.exists(scat_path):
                os.makedirs(scat_path)
            fig.savefig(os.path.join(
                scat_path, 'brain_pca_' + str(idx+1).zfill(2) + '.png'), dpi=160)

    for idx in range(brain_pca_comps.shape[1]):
        plot_vol_stc_brainmap(save_path, 'brain_pca_' + str(idx+1).zfill(2), 
                              brain_pca_mixing[:, idx], vertices, '10', subjects_dir)

    plt.rcParams.update({'font.size': 20.0})
    plt.rcParams.update({'lines.linewidth': 3.0})

    if save_path:
        colorbar_path = os.path.join(save_path, 'colorbars')
        if not os.path.exists(colorbar_path):
            os.makedirs(colorbar_path)
    fig, ax = plt.subplots()

    cmap = plt.cm.RdBu_r
    cbl = mpl.colorbar.ColorbarBase(ax, cmap=cmap, ticks=[0, 1], 
                                    orientation='vertical')
    cbl.ax.set_yticklabels(['–', '+'])
    cbl.ax.set_aspect(8)
 
    if save_path:
        fig.savefig(os.path.join(colorbar_path, 'pca.png'), dpi=150)

    ## BRAIN PCASSO

    pca_params = {
        'n_components': n_brain_components,
    }

    pcasso = Icasso(PCA, ica_params=pca_params, iterations=data.shape[0],
                    bootstrap=True, vary_init=False)

    counter = 0
    def bootstrap_fun(data, generator):
        global counter
        indices = [val for val in range(data.shape[0])]; del indices[counter]
        counter += 1
        return data[indices, :]

    def unmixing_fun(pca):
        return pca.components_

    fit_params = {}

    pcasso.fit(data=data, fit_params=fit_params, bootstrap_fun=bootstrap_fun,
               unmixing_fun=unmixing_fun)

    if save_path:
        pcasso_path = os.path.join(save_path, 'pcasso')
        if not os.path.exists(pcasso_path):
            os.makedirs(pcasso_path)

    plt.rcParams.update({'font.size': 30.0})
    plt.rcParams.update({'lines.linewidth': 4.0})

    print("Plotting dendrogram")
    if save_path:
        fig = pcasso.plot_dendrogram(show=False)
        path = os.path.join(pcasso_path, 'brain_dendrogram.png')
        fig.gca().set_xlabel('')
        fig.savefig(path, dpi=60)
    else:
        pcasso.plot_dendrogram()

    print("Plotting mds")
    if save_path:
        fig = pcasso.plot_mds(show=False)
        path = os.path.join(pcasso_path, 'brain_mds.png')
        fig.savefig(path, dpi=60)
    else:
        pcasso.plot_mds(show=False)

    brain_pcasso_unmixing, brain_pcasso_scores = (
        pcasso.get_centrotype_unmixing(distance=0.7))

    for idx in range(n_brain_components):
        if np.mean(brain_pcasso_unmixing[idx, :]) < 0:
            brain_pcasso_unmixing[idx, :] = -brain_pcasso_unmixing[idx, :]

    print("Brain passo scores")
    print(str(brain_pcasso_scores))

    print("Plotting brain pcasso")
    for idx in range(n_brain_components):
        plot_vol_stc_brainmap(save_path, 'brain_pcasso_' + str(idx+1).zfill(2), 
                              brain_pcasso_unmixing[idx, :], vertices, '10', subjects_dir)

    ## regression

    plt.rcParams.update({'font.size': 10.0})
    plt.rcParams.update({'lines.linewidth': 3.0})

    if save_path:
        stat_path = os.path.join(save_path, 'stats')
        if not os.path.exists(stat_path):
            os.makedirs(stat_path)


    # set sns style.. hard to set off
    sns.set(style='ticks', font_scale=1.2, font='Arial')

    x = brain_pca_comps[:, 0]
    y = behav_comps[:, 0]
    frame = pd.DataFrame(np.transpose([x, y]), columns=['Brain component', 'Behavioral component'])
    grid = sns.jointplot(x='Brain component', y='Behavioral component', data=frame, kind='reg',
                         )
    if save_path:
        grid.savefig(os.path.join(stat_path, 'regression_joint.png'), dpi=155)

    fig, ax = plt.subplots()
    sns.regplot(x='Brain component', y='Behavioral component', data=frame, ax=ax,
                )
    if save_path:
        fig.savefig(os.path.join(stat_path, 'regression.png'), dpi=155)

    fig, ax = plt.subplots()
    sns.residplot(x='Brain component', y='Behavioral component', data=frame, ax=ax,
                  )
    if save_path:
        fig.savefig(os.path.join(stat_path, 'resid.png'), dpi=155)

    X = x.copy()
    Y = y.copy()
    model = sm.OLS(Y, X)
    results = model.fit()
    print(results.summary())

    ## plot detailed process of pca

    subject_values = []
    for idx in range(28):
        subject_map = data[idx, :] - brain_pca.mean_
        result = brain_pca_mixing[:, 0] * subject_map
        subject_values.append(np.mean(result))

    plt.rcParams.update({'font.size': 8.0})
    plt.rcParams.update({'lines.linewidth': 3.0})

    y = [np.random.normal() for idx in range(len(subject_values))] 
    for idx in range(len(subject_values)):
        subject_map = data[idx, :] - brain_pca.mean_
        result = brain_pca_mixing[:, 0] * subject_map

        fig, axes = plt.subplots(nrows=1, ncols=2)

        plot_vol_stc_brainmap(None, '', data[idx, :], vertices, '10', 
                              subjects_dir, axes=axes[0])

        y_distance = np.max(y) - np.min(y)
        axes[1].set_ylim([np.min(y) - 3.0*y_distance, 
                          np.max(y) + 3.0*y_distance])

        axes[1].scatter(subject_values, y, color='b', s=13)
        axes[1].scatter(subject_values[idx], y[idx], color='r', s=13)
        axes[1].set_yticklabels([])
        # axes[1].set_xlim([-0.012, 0.012])
        axes[1].set_aspect(0.0005)

        if save_path:
            ind_path = os.path.join(save_path, 'individuals')
            if not os.path.exists(ind_path):
                os.makedirs(ind_path)
            fig.savefig(
                os.path.join(ind_path, 
                             'subject_'  + str(idx+1).zfill(2) + '.png'), 
                dpi=310)

    ## ICA

    from sklearn.decomposition import FastICA
    brain_ica = FastICA(n_components=n_brain_components, max_iter=10000)
    brain_ica_comps = brain_ica.fit_transform(data)

    factors = np.std(brain_ica_comps, axis=0)
    brain_ica_comps = np.divide(brain_ica_comps, factors)
    brain_ica_mixing = np.multiply(np.linalg.pinv(brain_ica.components_), factors)
    brain_ica_unmixing = np.divide(brain_ica.components_.T, factors).T

    # must build this again
    # total_variance = np.sum(np.diagonal(np.cov(data.T)))
    # sum_ = 0
    # for idx in range(n_brain_components):
    #     comp_variance = (np.sum(brain_ica_mixing[:, idx]**2) *
    #                      np.var(brain_ica_comps[:, idx]))
    #     sum_ += comp_variance
    #     print("ica brain " + str(idx+1) + ", var explained: " + 
    #           str(comp_variance / total_variance))
    # print("Sum: " + str(sum_ / total_variance))

    ica_vars = np.var(brain_ica_mixing, axis=0)
    ica_vars = ica_vars / np.sum(ica_vars)
    ica_vars = ica_vars * np.sum(pca_explained_variance)
    print("ICA explained vars: " + str(ica_vars))

    import pdb; pdb.set_trace()

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

    for idx in range(n_brain_components):
        plot_vol_stc_brainmap(save_path, 'brain_ica_' + str(idx+1).zfill(1), 
                              brain_ica_mixing[:, idx], vertices, '10', subjects_dir)

    for brain_idx in range(n_brain_components):
        for behav_idx in range(n_behav_components):
            title = 'ica brain ' + str(brain_idx+1) + ', behav ' + str(behav_idx+1) + ': '
            pearson = scipy.stats.pearsonr(brain_ica_comps[:, brain_idx], 
                                           behav_comps_wh[:, behav_idx])
            print(title + str(pearson))

    for idx in range(n_brain_components):
        title = 'brain ' + str(brain_idx+1) + ' t-test: '
        result = scipy.stats.ttest_1samp(brain_ica_comps[:, idx] - baselines[idx], 0)
        print(title + str(result))

    raise Exception('Kissa')

    ## SPATIAL PCA

    brain_pca_spatial = PCA(n_components=n_brain_components, whiten=False)
    brain_spatial_comps = brain_pca_spatial.fit_transform(data.T) 
    orig_total_variance = np.sum(np.diag(np.cov(data)))
    try:
        new_vars = np.diag(np.cov(brain_spatial_comps.T))
    except:
        new_vars = np.var(brain_spatial_comps)
    print("Explained spatial pca brain var: " + str(new_vars/orig_total_variance))
    print("Sum: " + str(np.sum(new_vars/orig_total_variance)))

    brain_pca_spatial_wh = PCA(n_components=n_brain_components, whiten=True)
    brain_pca_spatial_comps_wh = brain_pca_spatial_wh.fit_transform(data.T) 
    brain_pca_spatial_mixing = np.linalg.pinv(brain_pca_spatial_wh.components_)

    for idx in range(n_brain_components):
        plot_vol_stc_brainmap(save_path, 'brain_pca_spatial_' + str(idx+1).zfill(1), 
                              brain_pca_spatial_comps_wh[:, idx], vertices, '10', subjects_dir)

    for brain_idx in range(n_brain_components):
        for behav_idx in range(n_behav_components):
            title = 'brain pca spatial ' + str(brain_idx+1) + ', behav ' + str(behav_idx+1) + ': '
            pearson = scipy.stats.pearsonr(brain_pca_spatial_mixing[:, brain_idx], 
                                           behav_comps_wh[:, behav_idx])
            print(title + str(pearson))

    for idx in range(n_brain_components):
        title = 'brain ' + str(idx+1) + ' t-test: '
        result = scipy.stats.ttest_1samp(brain_pca_spatial_mixing[:, idx], 0)
        print(title + str(result))

    ## SPATIAL ICA

    from sklearn.decomposition import FastICA
    brain_ica_spatial = FastICA(n_components=n_brain_components, max_iter=10000)
    brain_ica_spatial_comps = brain_ica_spatial.fit_transform(data.T)
    brain_ica_spatial_mixing = np.linalg.pinv(brain_ica_spatial.components_)

    for idx in range(n_brain_components):
        plot_vol_stc_brainmap(save_path, 'brain_ica_spatial_' + str(idx+1).zfill(1), 
                              brain_ica_spatial_comps[:, idx], vertices, '10', subjects_dir)

        # mean = brain_ica_spatial.mean_
        # baseline = brain_ica.transform(np.zeros(brain_ica_mixing.shape[0])[np.newaxis, :])

    total_variance = np.sum(np.diagonal(np.cov(data)))
    sum_ = 0
    for idx in range(n_brain_components):
        comp_variance = (np.sum(brain_ica_spatial_mixing[:, idx]**2) *
                         np.var(brain_ica_spatial_comps[:, idx]))
        sum_ += comp_variance
        print("spatial ica brain " + str(idx+1) + ", var explained: " + 
              str(comp_variance / total_variance))
    print("Sum: " + str(sum_ / total_variance))

    for brain_idx in range(n_brain_components):
        for behav_idx in range(n_behav_components):
            title = 'spatial ica brain ' + str(brain_idx+1) + ', behav ' + str(behav_idx+1) + ': '
            pearson = scipy.stats.pearsonr(brain_ica_spatial_mixing[:, brain_idx], 
                                           behav_comps_wh[:, behav_idx])
            print(title + str(pearson))

    for idx in range(n_brain_components):
        title = 'brain ' + str(brain_idx+1) + ' t-test: '
        result = scipy.stats.ttest_1samp(brain_ica_spatial_mixing[:, idx], 0)
        print(title + str(result))

    raise Exception('Kissa')

