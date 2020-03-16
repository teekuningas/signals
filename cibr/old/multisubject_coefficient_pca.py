PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('font', size=40.0)
    matplotlib.rc('lines', linewidth=5.0)
    matplotlib.use('Agg')

import matplotlib.pyplot as plt 
import matplotlib as mpl
plt.rcParams.update({'figure.max_open_warning': 0})

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

from nilearn.plotting import plot_glass_brain

from sklearn.decomposition import PCA

from scipy.signal import hilbert
from scipy.signal import decimate

from icasso import Icasso

from signals.cibr.common import preprocess
from signals.cibr.common import plot_vol_stc_brainmap


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
        fig_.savefig(path, dpi=620)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path')
    parser.add_argument('--coefficients', nargs='+')
    parser.add_argument('--info', nargs='+')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    names = [fname.split('/')[-1].split('.csv')[0] for fname in cli_args.info]

    data = []
    vertex_list = []
    info = []

    for fname in sorted(cli_args.coefficients):
       with open(fname, 'r') as f:
            lines = f.readlines()
            vertex_list.append([int(val) for val in lines[0].strip().split(', ')])
            data.append([float(val) for val in lines[1].split(', ')])

    data = np.array(data)
    vertex_list = np.array(vertex_list)

    for fname in sorted(cli_args.info):
        with open(fname, 'r') as f:
            lines = f.readlines()
            vals = lines[0].split(', ')
            info.append((vals[0], 
                         float(vals[1]), 
                         float(vals[2]), 
                         float(vals[3]), 
                         float(vals[4]))) 

    not_consistent = [idx for idx in range(len(info))
                      if info[idx][4] < 0.2]
    print("Not consistent: " + str(not_consistent))

    # not_accurate = [idx for idx in range(len(info))
    #                 if info[idx][3] < 0.54]
    # block2 good accuracy: 002, 004, 005, 006, 008, 009, 010, 013, 
    # 014, 015, 016, 017, 018, 019, 020, 021, 023, 024, 025, 027, 028
    not_accurate = [001, 007, 011, 012, 022, 026, 029]
    print("Not accurate: " + str(not_accurate))

    corrs = np.array([line[4] for line in info])

    print("Corr mean before: " + str(np.mean(corrs)))
    print("Corr min: " + str(np.min(corrs)))
    print("Corr max: " + str(np.max(corrs)))

    filt_corrs = np.array([line[4] for idx, line in enumerate(info) if 
                  idx not in not_consistent])
    print("Corr mean after: " + str(np.mean(filt_corrs)))

    # not_valid = list(set(not_consistent).union(set(not_accurate)))
    not_valid = not_accurate

    print("Left: " + str(len(info) - len(not_valid)))

    if save_path:
        for subj_idx in range(data.shape[0]):
            if subj_idx in not_valid:
                continue
            vertices = vertex_list[subj_idx]
            brainmap = data[subj_idx]
            savename = info[subj_idx][0] + '_brainmap'
            plot_vol_stc_brainmap(save_path, savename, brainmap,
                                  vertices, '10', subjects_dir)
            
    data = np.array([row for row_idx, row in enumerate(data)
                     if row_idx not in not_valid])

    names = [name for idx, name in enumerate(names) if idx not in not_valid]

    for idx in range(data.shape[0]):
        max_ = np.max(np.abs(data[idx]))
        print(names[idx] + ' max: ' + str(max_))
        data[idx] /= max_

    pca_params = {
        'n_components': 3,
    }

    pcasso = Icasso(PCA, ica_params=pca_params, iterations=1000,
                    bootstrap=True, vary_init=False)

    def bootstrap_fun(data, generator):
        # return data[generator.choice(data.shape[0], size=data.shape[0]), :]
        return data[generator.permutation(data.shape[0])[:data.shape[0]-1], :]

    def unmixing_fun(pca):
        return pca.components_

    fit_params = {}

    pcasso.fit(data=data, fit_params=fit_params, bootstrap_fun=bootstrap_fun,
               unmixing_fun=unmixing_fun)

    if not save_path:
        pcasso.plot_dendrogram()
        pcasso.plot_mds()
    else:

        pcasso_path = os.path.join(save_path, 'pcasso')
        if not os.path.exists(pcasso_path):
            os.makedirs(pcasso_path)

        fig = pcasso.plot_dendrogram(show=False)
        path = os.path.join(pcasso_path, 'dendrogram.png')
        fig.savefig(path, dpi=620)

        fig = pcasso.plot_mds(show=False)
        path = os.path.join(pcasso_path, 'mds.png')
        fig.savefig(path, dpi=620)

    c_unmixing, c_scores = pcasso.get_centrotype_unmixing(distance=0.7)
    print("Cluster index scores: " + str(c_scores))

    pca_comps = np.matmul(c_unmixing, data.T).T
    baseline = np.matmul(c_unmixing, np.zeros((data.shape[1], 1))).T
    pca_comps = pca_comps - baseline
    mixing = np.linalg.pinv(c_unmixing)

    # print("Fitting normal PCA")
    # pca = PCA(n_components=3)

    # pca_comps = pca.fit_transform(data) 
    # mixing = np.linalg.pinv(pca.components_)

    # baseline = pca.transform(np.zeros((1, mixing.shape[0])))
    # pca_comps = pca_comps - baseline

    orig_total_variance = np.sum(np.diag(np.cov(data.T)))
    new_vars = np.diag(np.cov(pca_comps.T))
    print("Explained vars: " + str(new_vars/orig_total_variance))

    # sort by explained variance
    print("Sorting comps according to explained variance")
    mixing = mixing[:, np.argsort(-new_vars)[:3]]
    pca_comps = pca_comps[:, np.argsort(-new_vars)[:3]]

    print("Plotting")
    vertices = vertex_list[0]
    for comp_idx in range(pca_comps.shape[1]):
        comp = pca_comps[:, comp_idx]
        difference_brainmap = mixing[:, comp_idx]
        plot_annot_points(save_path, 'comp_distr_' + str(comp_idx+1), 
                          comp, names)
        plot_vol_stc_brainmap(save_path, 'normal_pca_' + str(comp_idx+1).zfill(2), 
                              difference_brainmap, vertices, '10', subjects_dir)
        plot_brainmap_and_comps(save_path, 'comp_' + str(comp_idx+1),
                                comp, difference_brainmap, vertices, '10',
                                subjects_dir)

    if save_path:
        comp_path = os.path.join(save_path, 'comps.csv')
        with open(comp_path, 'w') as f:
            f.write('; '.join(names) + '\n')
            for comp_idx, comp in enumerate(pca_comps.T):
                f.write('; '.join([str(elem) for elem in comp]) + '\n')

    raise Exception('Miau')

