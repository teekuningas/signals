PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('font', size=8.0)
    matplotlib.use('Agg')

import matplotlib.pyplot as plt 
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

# from icasso import Icasso

from signals.cibr.common import preprocess


def plot_vol_stc_brainmap_multiple(save_path, name, brainmap_inc, brainmap_dec, vertices, spacing, subjects_dir):

    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    fig_ = plt.figure()

    brainmap_inc = (brainmap_inc - np.mean(brainmap_inc)) / np.std(brainmap_inc)
    brainmap_dec = (brainmap_dec - np.mean(brainmap_dec)) / np.std(brainmap_dec)

    stc_inc = mne.source_estimate.VolSourceEstimate(
        brainmap_inc[:, np.newaxis],
        vertices,
        tstep=0.1,
        tmin=0,
        subject='fsaverage')
    stc_dec = mne.source_estimate.VolSourceEstimate(
        brainmap_dec[:, np.newaxis],
        vertices,
        tstep=0.1,
        tmin=0,
        subject='fsaverage')

    src_fname = os.path.join(subjects_dir, 'fsaverage', 'bem', 
                             'fsaverage-vol-' + spacing + '-src.fif')
    src = mne.source_space.read_source_spaces(src_fname)

    t1_fname = os.path.join(subjects_dir, 'fsaverage', 'mri', 'aseg.mgz')
    t1_img = nib.load(t1_fname)

    nifti_inc = stc_inc.as_volume(src).slicer[:, :, :, 0]
    nifti_dec = stc_dec.as_volume(src).slicer[:, :, :, 0]

    display = plot_glass_brain(t1_img, figure=fig_, display_mode='lyrz')
    display.add_overlay(nifti_inc, alpha=0.9, cmap='Reds')
    display.add_overlay(nifti_dec, alpha=0.5, cmap='Blues')
    if not save_path:
        plt.show()

    if save_path:

        brain_path = os.path.join(save_path, 'vol_brains')
        if not os.path.exists(brain_path):
            os.makedirs(brain_path)

        path = os.path.join(brain_path,
            name + '.png')

        fig_.savefig(path, dpi=620)


def plot_vol_stc_brainmap(save_path, name, brainmap, cmap, vertices, spacing, subjects_dir):

    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    fig_ = plt.figure()

    stc = mne.source_estimate.VolSourceEstimate(
        brainmap[:, np.newaxis],
        vertices,
        tstep=0.1,
        tmin=0,
        subject='fsaverage')

    src_fname = os.path.join(subjects_dir, 'fsaverage', 'bem', 
                             'fsaverage-vol-' + spacing + '-src.fif')
    src = mne.source_space.read_source_spaces(src_fname)

    t1_fname = os.path.join(subjects_dir, 'fsaverage', 'mri', 'aseg.mgz')
    t1_img = nib.load(t1_fname)

    nifti = stc.as_volume(src).slicer[:, :, :, 0]

    display = plot_glass_brain(t1_img, figure=fig_, display_mode='lyrz')
    # display.add_overlay(nifti, alpha=0.75)
    display.add_overlay(nifti, alpha=0.75, cmap=cmap)

    if not save_path:
        plt.show()

    if save_path:

        brain_path = os.path.join(save_path, 'vol_brains')
        if not os.path.exists(brain_path):
            os.makedirs(brain_path)

        path = os.path.join(brain_path,
            name + '.png')

        fig_.savefig(path, dpi=620)


def plot_annot_points(save_path, name, points, annots, weights):

    fig_, ax = plt.subplots()

    bins = np.linspace(np.min(points), np.max(points), 50)

    y = []
    labels = []
    for idx in range(len(bins)-1):
        mask = np.where((points >= bins[idx]) & (points < bins[idx+1]))
        if np.any(mask):
            labels.append([name for name in np.array(annots)[mask]])
            y.append(bins[idx])

    x = [0]*len(y)

    bin_weights = [len(labels[idx]) for idx in range(len(y))] 
    allvals = []
    for idx, val in enumerate(y):
       allvals.extend([val]*bin_weights[idx])
    median = np.median(allvals)
    
    if weights is not None:
        plt.scatter(x, y, c=weights, cmap='OrRd')
    else:
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

    if not save_path:
        plt.show()

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
            info.append((vals[0], float(vals[1]), float(vals[2]), float(vals[3])))

    weights = []

    for idx in range(data.shape[0]):
        scores = [val[1] for val in info]
        # factor = (scores[idx] - np.min(scores)) / (np.max(scores) - np.min(scores))

        # think about this again
        # factor = np.sqrt((scores[idx] - 0.5) / 0.5)

        # factor = (scores[idx] - 0.5) / 0.25
        # if factor < 0: 
        #     factor = 0
        # if factor > 1:
        #     factor = 1
        
        # if scores[idx] > 0.57:
        #     factor = 1.0
        # else:
        #     factor = 0.0

        # if scores[idx] > 0.6:
        #     factor = 1.0
        # elif scores[idx] < 0.5:
        #     factor = 0.0
        # else:
        #     factor = np.sqrt((scores[idx] - 0.5) / 0.1)


        print(names[idx] + ' max: ' + str(np.max(np.abs(data[idx]))))
        print(names[idx] + ' mean: ' + str(np.mean(np.abs(data[idx]))))

        data[idx] /= np.max(np.abs(data[idx]))

        # factor = 1.0
        # weights.append(factor)
        # data[idx] *= factor
        # weights.append(1.0)
        # data[idx] *= 1.0

    print("Fitting normal PCA")
    pca = PCA(n_components=3)
    pca_comps = pca.fit_transform(data) 
    mixing = np.linalg.pinv(pca.components_)

    for idx in range(mixing.shape[1]):
        if np.mean(mixing[:, idx]) < 0:
            mixing[:, idx] = -mixing[:, idx]    
            pca_comps[idx] = -pca_comps[idx]

    for comp_idx in range(pca_comps.shape[1]):
        comp = pca_comps[:, comp_idx]
        # comp[comp > 0] = np.sqrt(comp[comp>0])
        # comp[comp < 0] = -np.sqrt(-comp[comp<0])
        plot_annot_points(save_path, 'comp_distr_' + str(comp_idx+1), 
                          comp, names, weights=None)

    print("Scores:")
    print(', '.join([str(val) for val in pca.explained_variance_ratio_]))

    print("Plotting")
    for idx, column in enumerate(mixing.T):
        difference_brainmap = column
        vertices = vertex_list[idx]
        increase_map = difference_brainmap.copy()
        increase_map[increase_map < 0] = 0
        decrease_map = difference_brainmap.copy()
        decrease_map[decrease_map > 0] = 0
        decrease_map = -decrease_map
        plot_vol_stc_brainmap_multiple(save_path, 'normal_pca_' + str(idx+1).zfill(2), increase_map, decrease_map, vertices, '10', subjects_dir) 

    if save_path:
        comp_path = os.path.join(save_path, 'comps.csv')
        with open(comp_path, 'w') as f:
            f.write('; '.join(names) + '\n')
            for comp_idx, comp in enumerate(pca_comps.T):
                f.write('; '.join([str(elem) for elem in comp]) + '\n')

    raise Exception('Miau')

