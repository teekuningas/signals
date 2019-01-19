PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('font', size=10)
    matplotlib.use('Agg')

import matplotlib.pyplot as plt 
plt.rcParams.update({'figure.max_open_warning': 0})

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

from signals.cibr.common import preprocess


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

    # mask = np.array([elem for elem in subject_info['MedExp'] == '1'])
    # pca_coefs = pca_coefs[:, mask]
    # background_data = background_data[mask, :]

    pca = PCA(n_components=2)
    tf_background = pca.fit_transform(background_data)
    mixing = np.linalg.pinv(pca.components_)

    import statsmodels.api as sm
    for comp in pca_coefs:
        Y = comp.copy()
        X = tf_background.copy()
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()
        print (results.summary())

    from scipy import stats
    for idx, comp in enumerate(pca_coefs):
        print('Comp ' + str(idx+1) + ': ')
        print(stats.ttest_1samp(comp, 0))

    raise Exception('Mijau, here starts the scatter plots')

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for header_name in ['BDI', 'BAI', 'BIS', 'BasDrive', 'BasRR', 'BasFS', 'MedLength', 'MedFreq', 'MedExp']:
            for comp_idx, comp in enumerate(pca_coefs):
                info = [float(value.replace(',', '.')) for value in subject_info[header_name].values]
                fig, ax = plt.subplots()
                fig.suptitle(header_name + ' x ' + 'comp ' + str(comp_idx+1))
                ax.scatter(comp, info)
                ax.set_ylabel(header_name)
                ax.set_xlabel('Value on principal component axis')
                fig.savefig(os.path.join(save_path, header_name.lower() + '_comp_' + str(comp_idx+1) + '.png'))
        # Med strategy
        header_name = 'MedStrategy2'
        for comp_idx, comp in enumerate(pca_coefs):
            info = [value for value in subject_info[header_name].values]
            info = [1 if value == 'openmonitoring' else value for value in info]
            info = [0 if value == 'focusedattention' else value for value in info]
            idxs = [idx for idx, val in enumerate(info) if val != '-']
            comp = np.array(comp)[idxs]
            info = [val for idx, val in enumerate(info) if val in idxs]

            fig, ax = plt.subplots()
            fig.suptitle(header_name + ' x ' + 'comp ' + str(comp_idx+1))
            ax.scatter(comp, info)
            ax.set_ylabel('Open monitoring (1) vs focused attention (0)')
            ax.set_xlabel('Value on principal component axis')

            fig.savefig(os.path.join(save_path, header_name.lower() + '_comp_' + str(comp_idx+1) + '.png'))

    raise Exception('Miau')

