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
from sklearn.decomposition import SparsePCA
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
    parser.add_argument('--drop')
    parser.add_argument('--identifier')
    parser.add_argument('--spacing')
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

    vol_spacing = '10'
    if cli_args.spacing is not None:
        vol_spacing = cli_args.spacing

    # meditaatio
    def name_from_fname(fname):
        return fname.split('/')[-1].split('_')[1]

    # # fdmsa
    # def name_from_fname(fname):
    #     if 'heart' in fname and 'note' in fname:
    #         return '_'.join(fname.split('/')[-1].split('_')[1:][:-2])
    #     else:
    #         return '_'.join(fname.split('/')[-1].split('_')[1:][:-1])

    # read contrast data from all participants
    drop_keys = cli_args.drop.split(' ') if cli_args.drop else []

    for fname in sorted(cli_args.coefficients_1):
        if any([key in fname for key in drop_keys]):
            continue

        with open(fname, 'r') as f:
            lines = f.readlines()
            vertex_list.append([val for val in lines[0].strip().split(', ')])
            data_1.append([float(val) for val in lines[1].split(', ')])
            names_1.append(name_from_fname(fname))
    data_1 = np.array(data_1)

    for fname in sorted(cli_args.coefficients_2):
        if any([key in fname for key in drop_keys]):
            continue

        with open(fname, 'r') as f:
            lines = f.readlines()
            data_2.append([float(val) for val in lines[1].split(', ')])
            names_2.append(name_from_fname(fname))
    data_2 = np.array(data_2)

    for fname in sorted(cli_args.coefficients_norm):
        if any([key in fname for key in drop_keys]):
            continue

        with open(fname, 'r') as f:
            lines = f.readlines()
            norm_data.append([float(val) for val in lines[1].split(', ')])
            names_norm.append(name_from_fname(fname))
    norm_data = np.array(norm_data)

    if not (names_1 == names_2 == names_norm):
        raise Exception('Names do not match')

    names = names_1

    contrast_data = np.mean([data_1, data_2], axis=0)

    # for plotting
    raw = mne.io.Raw(cli_args.example_raw, preload=True, verbose='warning')
    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names'])
                       if idx not in mne.pick_types(raw.info, meg=True)])

    pretransform = False

    if pretransform:
        print("Pretransforming non-normal variables")

        weights = np.abs(np.mean(norm_data, axis=0))
        weights = weights / np.max(weights)

        contrast_data = contrast_data * weights

    plt.rcParams.update({'font.size': 40.0})

    n_subjects = contrast_data.shape[0]

    fig = plt.figure()
    fig.suptitle('Individual spatial contrast maps', fontsize=150.0)

    fig.set_size_inches(35, 10*n_subjects)
    fig_dpi = 15

    # do for all subjects
    for subject_idx, subject_data in enumerate(contrast_data):

        ax_brain = plt.subplot2grid((n_subjects, 1), (subject_idx, 0), rowspan=1, colspan=1)

        # plot contrast part
        if len(subject_data) > 500:
            vertices = np.array([int(vx) for vx in vertex_list[0]])
            plot_vol_stc_brainmap(subject_data, vertices, vol_spacing, subjects_dir, ax_brain, cap=0.0)
        else:
            plot_sensor_topomap(subject_data, raw.info, ax_brain)

        name = names_1[subject_idx]

    if save_path:
        path = os.path.join(save_path, 'comps')
        if not os.path.exists(path):
            os.makedirs(path)
        fname = 'individual_spatial_contrast_maps_' + cli_args.identifier + '.png'
        fig.savefig(os.path.join(path, fname), dpi=fig_dpi)

    print("Done.")
