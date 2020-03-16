PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('font', size=5.0)
    matplotlib.rc('lines', linewidth=3.0)
    matplotlib.use('Agg')

import matplotlib.pyplot as plt 
import matplotlib as mpl
plt.rcParams.update({'figure.max_open_warning': 0})

import pyface.qt

import sys
import argparse
import os

from collections import OrderedDict

from statsmodels.multivariate.cancorr import CanCorr

import nibabel as nib
import mne
import numpy as np
import sklearn
import pandas as pd
import scipy.stats

from nilearn.plotting import plot_glass_brain

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

from scipy.signal import hilbert
from scipy.signal import decimate

from icasso import Icasso

from signals.cibr.common import preprocess
from signals.cibr.common import plot_vol_stc_brainmap

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def plot_brainmap_and_comps(save_path, name, points, brainmap, vertices, spacing,
                            subjects_dir):

    fig_ = plt.figure()

    # ax_brain = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    # ax_points = plt.subplot2grid((1, 3), (0, 2))

    ax_brain = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax_colorbar = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    ax_points = plt.subplot2grid((2, 3), (0, 2), rowspan=2)

    plot_vol_stc_brainmap(None, name, brainmap, vertices, 
                          spacing, subjects_dir, axes=ax_brain)

    y = points
    x = [0]*len(y)
    median = np.mean(y)
    
    ax_points.scatter(x, y, facecolors='none', edgecolors='black')

    ax_points.axhline(0, linewidth=2, color='r')
    ax_points.axhline(median, linewidth=1, color='g')
    ax_points.set_xlim(-0.25, 0.25)
    ax_points.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    # ax_points.set_yticks([0])
    # yticks = ax_points.get_yticks()
    # yticklabels = ['0' if np.allclose(tick, 0) else '' for tick in yticks]
    # yticklabels[0] = '-'
    # yticklabels[-1] = '+'
    # ax_points.set_yticklabels(yticklabels)

    cmap = plt.cm.RdBu_r
    cbl = mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cmap, ticks=[0,1], 
                                    orientation='horizontal')
    cbl.ax.set_xticklabels(['Low', 'High'])

    ax_points.set_position([
        ax_points.get_position().x0,
        ax_points.get_position().y0 + ax_points.get_position().height / 2.5,
        ax_points.get_position().width,
        ax_points.get_position().height / 2.0,
    ])

    ax_colorbar.set_position([
        ax_colorbar.get_position().x0,
        ax_colorbar.get_position().y0 + ax_colorbar.get_position().height,
        ax_colorbar.get_position().width / 1.1,
        ax_colorbar.get_position().height / 4.0,
    ])

    ax_brain.set_position([
        ax_brain.get_position().x0 - ax_brain.get_position().width / 8.0,
        ax_brain.get_position().y0,
        ax_brain.get_position().width,
        ax_brain.get_position().height,
    ])

    if save_path:
        comp_path = os.path.join(save_path, 'brainmap_comps')
        if not os.path.exists(comp_path):
            os.makedirs(comp_path)

        path = os.path.join(comp_path, name + '.png')
        fig_.savefig(path, dpi=155)

def plot_annot_points(save_path, name, points, annots):

    fig_, ax = plt.subplots()

    bins = np.linspace(np.min(points), np.max(points), 50)

    y = []
    labels = []
    for idx in range(len(bins)-1):
        mask = np.where((points >= bins[idx]) & (points <= bins[idx+1]))
        if np.size(mask) > 0:
            labels.append([name for name in np.array(annots)[mask]])
            y.append(bins[idx])

    x = [0]*len(y)

    bin_weights = [len(labels[idx]) for idx in range(len(y))] 
    allvals = []
    for idx, val in enumerate(y):
       allvals.extend([val]*bin_weights[idx])
    median = np.median(allvals)
    
    plt.scatter(x, y)

    for idx, xy in enumerate(zip(x, y)):
        ax.annotate('(%s)' % (', '.join(labels[idx])), xy=xy, textcoords='data')

    ax.axhline(0, linewidth=2, color='r')
    ax.axhline(median, linewidth=1, color='g')
    
    ax.set_xlim(-0.5, 4.0)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    # if not save_path:
    #     plt.show()

    if save_path:
        annot_path = os.path.join(save_path, 'annots')
        if not os.path.exists(annot_path):
            os.makedirs(annot_path)

        path = os.path.join(annot_path, name + '.png')
        fig_.savefig(path, dpi=620)


def plot_barcharts(save_path, components, savename, names):
    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    y_pos = np.arange(len(names))

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
        fig_.savefig(path, dpi=620)



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

    # n_brain_components = 10
    n_behav_components = 3

    for counter in range(5):
        n_brain_components = 5 - counter
        print("Run with " + str(n_brain_components) + " components.")

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

        vertices = vertex_list[0]

        brain_pca = PCA(n_components=n_brain_components, whiten=False)
        brain_pca_comps = brain_pca.fit_transform(data) 
        orig_total_variance = np.sum(np.diag(np.cov(data.T)))
        try:
            new_vars = np.diag(np.cov(brain_pca_comps.T))
        except:
            new_vars = np.var(brain_pca_comps)
        print("Explained pca brain var: " + str(new_vars/orig_total_variance))
        print("Sum: " + str(np.sum(new_vars/orig_total_variance)))

        brain_pca_wh = PCA(n_components=n_brain_components, whiten=True)
        brain_pca_comps_wh = brain_pca_wh.fit_transform(data) 
        brain_pca_mixing = np.linalg.pinv(brain_pca_wh.components_)

        plot_vol_stc_brainmap(save_path, 'first_pca', 
            brain_pca_mixing[:, 0], vertices, '10', subjects_dir)

        plot_barcharts(save_path, behav_mixing[:, 0], 'first_behav', 
                       behav_info_header[1:7])

        # cca = PCA(n_components=2)
        # cca_comps = cca.fit_transform(np.hstack([behav_comps_wh, brain_pca_comps_wh]))
        # cca_mixing = np.linalg.pinv(cca.components_)
        # cca_unmixing = cca.components_

        n_cca_components = 3

        cca = FastICA(n_components=n_cca_components)
        cca_comps = cca.fit_transform(np.hstack([behav_comps_wh, brain_pca_comps_wh]))
        cca_mixing = np.linalg.pinv(cca.components_)
        cca_unmixing = cca.components_

        for idx in range(n_cca_components):
            print("CCA component: " + str(idx+1))
            brain_weights = cca_mixing[n_behav_components:, idx]
            brain_unmixing = cca_unmixing[idx, n_behav_components:]
            brainmap = np.dot(brain_pca_mixing, brain_weights)

            behav_weights = cca_mixing[:n_behav_components, idx]
            behav_unmixing = cca_unmixing[idx, :n_behav_components]
            behavmap = np.dot(behav_mixing, behav_weights)

            x = np.dot(brain_pca_comps_wh, brain_unmixing)
            y = np.dot(behav_comps_wh, behav_unmixing)
            corrcoef = np.corrcoef(x, y)
            print("Correlation coefficient in CCA: " + str(corrcoef[0,1]))
            title = 'brainmap_' + str(n_behav_components) + '_' + str(n_brain_components) + '_' + str(idx)
            plot_vol_stc_brainmap(save_path, title, 
                                  brainmap, vertices, '10', subjects_dir)

            title = 'behav_weights_' + str(n_behav_components) + '_' + str(n_brain_components) + '_' + str(idx)
            plot_barcharts(save_path, behavmap, title, 
                           behav_info_header[1:7])

            plt.show()

