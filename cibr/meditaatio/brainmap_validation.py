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
import itertools

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


def get_grouping(paths):
    common_idxs = []
    codes = []
    for path in paths:
        codes.append('_'.join(path.split('/')[-1].split('_')[1:2]))

    paths_wout_codes = []
    for idx, path in enumerate(paths):
        folder_part = '/'.join(path.split('/')[0:-1])
        fname_part = path.split('/')[-1]
        paths_wout_codes.append(folder_part + '/' + fname_part.replace(codes[idx], ''))

    identifiers = []
    for path in paths_wout_codes:
        identifiers.append([item for subpath in path.split('/') for item in subpath.split('_')])

    differing_idxs = []
    for idx in range(len(identifiers[0])):
        all_same = True
        for sub1 in identifiers:
            for sub2 in identifiers:
                if sub1[idx] != sub2[idx]:
                    all_same = False
        if all_same:
            continue

        differing_idxs.append(idx)

    keys = []
    for idx in differing_idxs:
        vals = list(set([sub[idx] for sub in identifiers]))
        keys.append(vals)

    groups = {}
    for prod in itertools.product(*keys):
        group_key = prod
        sub_idxs = []
        for identifier_idx, identifier in enumerate(identifiers):
            all_true = True
            for prod_idx, prod_key in enumerate(prod):
                if prod_key != identifier[differing_idxs[prod_idx]]:
                    all_true = False
                    break
            if all_true:
                if group_key not in groups:
                    groups[group_key] = []

                groups[group_key].append(identifier_idx)

    return groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coefficients', nargs='+')
    parser.add_argument('--verbose')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    data = []
    vertex_list = []

    coefficients = sorted(cli_args.coefficients)

    for fname in sorted(coefficients):
       with open(fname, 'r') as f:
            lines = f.readlines()
            vertex_list.append([int(val) for val in lines[0].strip().split(', ')])
            data.append([float(val) for val in lines[1].split(', ')])

    data = np.array(data)
    vertex_list = np.array(vertex_list)

    # group into categories
    groups = get_grouping(coefficients)

    names = []
    for fname in coefficients:
        names.append('_'.join(fname.split('/')[-1].split('_')[1:2]))

    vertices = vertex_list[0]

    for pair in itertools.combinations(groups.keys(), 2):
        group_key_1 = pair[0]
        group_key_2 = pair[1]
        print("Pair: " + str(group_key_1) + ", " + str(group_key_2))
        subs_1 = groups[group_key_1]
        subs_2 = groups[group_key_2]
        all_corrcoefs = []
        for sub_idx_1 in subs_1:
            for sub_idx_2 in subs_2:
                data_1 = data[sub_idx_1]
                data_2 = data[sub_idx_2]

                corrcoef = np.corrcoef(data_1, data_2)[0, 1]

                if names[sub_idx_1] == names[sub_idx_2]:
                    all_corrcoefs.append(corrcoef)
                    if cli_args.verbose == 'true':
                        if corrcoef > 0.5:
                            print(names[sub_idx_1] + ": " + str(corrcoef) + " (HIGH!)")
                        elif corrcoef < 0.1:
                            print(names[sub_idx_1] + ": " + str(corrcoef) + " (LOW!)")
                        else:
                            print(names[sub_idx_1] + ": " + str(corrcoef))
                    break
        print("Average: " + str(np.mean(all_corrcoefs)))
        print("")

