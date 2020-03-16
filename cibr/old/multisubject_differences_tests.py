PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('font', size=12)
    matplotlib.use('Agg')

import matplotlib.pyplot as plt 
plt.rcParams.update({'figure.max_open_warning': 0})

# plt.style.use('seaborn-whitegrid')

import pyface.qt

import sys
import csv
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

from scipy import stats


def plot_barcharts(save_path, components, savename, names):
    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    y_pos = np.arange(len(names))
    for idx, component in enumerate(components):
        print("Plotting barchart")

        fig_, ax = plt.subplots()

        ax.bar(y_pos, component, align='center', alpha=0.5)
        ax.set_xticks(y_pos)
        ax.set_xticklabels(names)
        ax.set_ylabel('PCA weights')

        if not save_path:
            plt.show()

        if save_path:
            weight_path = os.path.join(save_path, 'weights')
            if not os.path.exists(weight_path):
                os.makedirs(weight_path)
            name = savename + '_' + str(idx+1).zfill(2) + '.png'

            path = os.path.join(weight_path, name)
            fig_.savefig(path, dpi=620)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path')
    parser.add_argument('--subject_info')
    parser.add_argument('--pca_coefs')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    pca_coefs = []
    with open(cli_args.pca_coefs, 'r') as f:
        lines = f.readlines()
        pca_coefs_header = lines[0].split(';')
        for line in lines[1:]:
            pca_coefs.append([float(val) for val in line.split('; ')])
    pca_coefs = np.array(pca_coefs)

    subject_info = []
    with open(cli_args.subject_info, 'r') as f:
        lines = f.readlines()
        subject_info_header = ['id'] + [elem.strip('\n') for elem in lines[0][1:].split(';')]
        for line in lines[1:]:
            elems = [elem.strip('\n') for elem in line.split(';')]
            subject_info.append([elems[0].zfill(3)] + elems[1:])

    subject_info = pd.DataFrame(subject_info, columns=subject_info_header)

    subject_ids = [name.split('_')[1] for name in pca_coefs_header]
    subject_info = subject_info.loc[subject_info['id'].isin(subject_ids)]

    background_data = subject_info.iloc[:, 1:7].values.astype(np.float)

    pca = PCA(n_components=1)
    tf_background = pca.fit_transform(background_data)
    mixing = np.linalg.pinv(pca.components_)

    print("Scores:")
    print(', '.join([str(val) for val in pca.explained_variance_ratio_]))

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path) 

    import statsmodels.api as sm
    for comp_idx, comp in enumerate(pca_coefs):
        Y = comp.copy()
        X = tf_background.copy()
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()
        print (results.summary())
        if save_path:
            log_path = os.path.join(save_path, 'logs')
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            log_fname = 'regr_comp_' + str(comp_idx+1) + '.txt'
            with open(os.path.join(log_path, log_fname), 'w') as f:
                f.write(str(results.summary()))
        
        if save_path:
            scatter_path = os.path.join(save_path, 'scatters')
            if not os.path.exists(scatter_path):
                os.makedirs(scatter_path)

        for bg_idx in range(tf_background.shape[1]):
            X = tf_background[:, bg_idx]
            Y = comp
            fig, ax = plt.subplots()
            title = ''
            # if results.f_pvalue < 0.05:
            #     title = 'p = ' + str(round(results.f_pvalue, 3))
            fig.suptitle(title)
            ax.scatter(X, Y)
            ax.set_ylabel('Brain PCA component')
            ax.set_xlabel('Psychometric PCA component')

            line_x = np.linspace(np.min(X), np.max(X), len(X))
            ax.plot(line_x, line_x*results.params[1] + results.params[0]) 

            savename = ('bg_comp_' + str(bg_idx+1) + '_brain_comp_' + 
                        str(comp_idx+1))
            if save_path:
                fig.savefig(os.path.join(scatter_path, savename + '.png'), 
                            dpi=620)
            else:
                plt.show()

    # from scipy import stats
    # for comp_idx, comp in enumerate(pca_coefs):
    #     message = ("Comp " + str(comp_idx+1) + ":\n" + 
    #                str(stats.ttest_1samp(comp, 0)) + "\n")
    #     print(message)
    #     if save_path:
    #         log_path = os.path.join(save_path, 'logs')
    #         if not os.path.exists(log_path):
    #             os.makedirs(log_path)
    #         log_fname = 'ttest_comp_' + str(comp_idx+1) + '.txt'
    #         with open(os.path.join(log_path, log_fname), 'w') as f:
    #             f.write(message)

    # mixing
    names = subject_info.iloc[:, 1:7].columns.values
    plot_barcharts(save_path, mixing.T, 'bg', names)

    print("Hooray, finished")

