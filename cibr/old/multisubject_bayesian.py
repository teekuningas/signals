PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('font', size=10)
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

from sklearn.decomposition import PCA

import pymc3 as pm


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

    # do bayesian regression

    with pm.Model() as model:
        alpha = pm.Normal('Intercept', mu=0, sd=10)
        beta = pm.Normal('Coefficients', mu=0, sd=10, shape=3)

        # sigma = pm.HalfNormal('sigma', sd=10)
        sigma = pm.HalfCauchy('Standard deviation', beta=10)

        mu = alpha + beta[0]*pca_coefs[0] + beta[1]*pca_coefs[1] + \
            beta[2]*pca_coefs[2]

        Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, 
                          observed=tf_background[:, 0])

        trace = pm.sample(2000, cores=2, chains=2, tune=1000)

    plt.clf(); pm.plots.traceplot(trace, combined=True); plt.show()
    if save_path:
        plt.gcf().savefig(os.path.join(save_path, 'regression_trace.png'), dpi=620)
        with open(os.path.join(save_path, 'regression_summary.txt'), 'w') as f:
            f.write(str(pm.stats.summary(trace)))

    # do bayesian confidence regions

    with pm.Model() as model:
        comp_mean = pm.Normal('Mean', np.mean(pca_coefs),
                              np.std(pca_coefs)*5, shape=pca_coefs.shape[0])
        comp_std = pm.HalfCauchy('Standard deviation', beta=10, shape=pca_coefs.shape[0])
        comp_prec = comp_std**-2
        v = pm.Exponential('v_minus_one', 1/29.) + 1

        comp = pm.StudentT('comp', nu=v, mu=comp_mean, lam=comp_prec,
                           observed=pca_coefs.T, shape=pca_coefs.shape[0])

        trace = pm.sample(2000, cores=2, chains=2, tune=1000)

    plt.clf(); pm.plots.traceplot(trace, combined=True, varnames=['Mean', 'Standard deviation']); plt.show()

    if save_path:
        plt.gcf().savefig(os.path.join(save_path, 'comp_trace.png'), dpi=620)
        with open(os.path.join(save_path, 'distr_summary.txt'), 'w') as f:
            f.write(str(pm.stats.summary(trace)))

    if not save_path:
        import pdb; pdb.set_trace() 
        print("miau")
