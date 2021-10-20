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
    parser.add_argument('--postica')
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

    if cli_args.postica == 'true':
        postica = True
    else:
        postica = False

    random_state = 15
    n_contrast_components = 3
    behav_vars = ['BDI', 'BAI', 'BIS', 'BasTotal']

    behav_data = []
    for name in names:
        row = []
        for var_name in behav_vars:
            row.append(float(questionnaire[questionnaire['id'] == name][var_name].values[0]))
        behav_data.append(row)

    behav_data = np.array(behav_data)

    contrast_pca = PCA(n_components=n_contrast_components, whiten=True)
    contrast_wh = contrast_pca.fit_transform(np.array(contrast_data))

    print("Explained variance: " + str(np.sum(contrast_pca.explained_variance_ratio_)))

    if not postica:
        contrast_mixing = contrast_pca.components_
        contrast_exp_var = contrast_pca.explained_variance_ratio_
    else:
        contrast_ica = FastICA(whiten=False, random_state=random_state)
        contrast_wh = contrast_ica.fit_transform(contrast_wh)
        contrast_mixing = np.dot(contrast_ica.components_, contrast_pca.components_)
        contrast_exp_var = ((np.var(contrast_mixing, axis=1) / 
                             np.sum((np.var(contrast_mixing, axis=1)))) * 
                            np.sum(contrast_pca.explained_variance_ratio_))

    for contrast_idx in range(n_contrast_components):
        """
        """
        for behav_idx, behav_name in enumerate(behav_vars):
            """
            """
            contrast_weights = contrast_mixing[contrast_idx]

            X = contrast_wh[:, contrast_idx]
            Y = behav_data[:, behav_idx]

            pearson_coef, pearson_pvalue = scipy.stats.pearsonr(X, Y)
            spearman_coef, spearman_pvalue = scipy.stats.spearmanr(X, Y)

            exp_var = contrast_exp_var[contrast_idx]

            fig, ax = plt.subplots(3)

            # plot contrast part
            if len(contrast_weights) > 500:
                vertices = np.array([int(vx) for vx in vertex_list[0]])
                plot_vol_stc_brainmap(contrast_weights, vertices, '10', subjects_dir, ax[1])
            else:
                plot_sensor_topomap(contrast_weights, raw.info, ax[1])

            frame = pd.DataFrame(np.transpose([X, Y]),
                                 columns=['Brain component', 'Behavioral component'])

            sns.regplot(x='Brain component', y='Behavioral component',
                    data=frame, ax=ax[2], scatter_kws={'s': 5}, line_kws={'lw': 1})

            decomp_type = 'ICA' if postica else 'PCA'

            title = '{0} component {1}\nExp var: {2:.3f}\nPearson: {3:.3f} ({4:.3f})\nSpearman: {5:.3f} ({6:.3f})'.format(
                decomp_type, str(contrast_idx+1).zfill(1), exp_var,
                pearson_coef, pearson_pvalue, 
                spearman_coef, spearman_pvalue)

            ax[0].axis('off')
            fig.suptitle(title)

            if save_path:
                path = os.path.join(save_path, 'results')
                if not os.path.exists(path):
                    os.makedirs(path)
                fname = (decomp_type.lower() + '_' + str(cli_args.identifier) + '_' + 
                         str(contrast_idx+1).zfill(2) + '_' + behav_name.lower() + '.png')
                fig.savefig(os.path.join(path, fname))

    print("Finished.")

