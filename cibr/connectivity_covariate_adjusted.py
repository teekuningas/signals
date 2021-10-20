import argparse
import os
import sys
import time


from pprint import pprint

import matplotlib
matplotlib.use('Agg')
matplotlib.rc('figure', max_open_warning=0)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import statsmodels.api as sm
import scipy.stats

import mne
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from scipy.sparse import coo_matrix

import seaborn as sns
import pandas as pd

import scipy.fftpack as fftpack
from scipy.signal import hilbert
from scipy.signal import decimate

from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

from signals.cibr.lib.stc import plot_vol_stc_brainmap

from hpica import HPICA

def fast_hilbert(x):
    return hilbert(x, fftpack.next_fast_len(x.shape[-1]))[..., :x.shape[-1]]

def save_fig(fig, path, dpi):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if os.path.isfile(path):
        os.remove(path)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

parser = argparse.ArgumentParser()
parser.add_argument('--raw', nargs='+')
parser.add_argument('--med_recon')
parser.add_argument('--ic_recon')
parser.add_argument('--n_components')
parser.add_argument('--decimation')
parser.add_argument('--questionnaire')
parser.add_argument('--empty', nargs='+')
parser.add_argument('--save_path')

cli_args = parser.parse_args()

random_state = 10

prefilter_band = (1, 40)
band = (7, 14)

# prefilter_band = (0.1, 40)
# band = (2, 4)

# prefilter_band = (1, 40)
# band = (20, 30)

sfreq = 100
vol_spacing = '10'

n_components = 20 
if cli_args.n_components:
    n_components = int(cli_args.n_components)

decimation_factor = 10
if cli_args.decimation:
    decimation_factor = int(cli_args.decimation)


print("Reading questionnaire data")
questionnaire = []
questionnaire_names = []
with open(cli_args.questionnaire, 'r') as f:
    lines = f.readlines()
    questionnaire_header = lines[0].strip().split(',')
    for line in lines[1:]:
        questionnaire.append(line.strip().split(',')[1:])
        name = line.strip().split(',')[0]
        if name.startswith('M'):
            questionnaire_names.append('subject_' + name[1:])
        elif name.startswith('I'):
            questionnaire_names.append('IC_' + name[1:])
        else:
            raise Exception('invalid name')


subjects_dir = cli_args.med_recon or cli_args.ic_recon

print("Computing noise covariance..")
empty_paths = cli_args.empty
empty_raws = []
for path in empty_paths:
    raw = mne.io.read_raw_fif(path, preload=True, verbose='error')
    raw.filter(*prefilter_band)
    raw.crop(tmin=(raw.times[0]+3), tmax=raw.times[-1]-3)
    raw.resample(sfreq)
    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names'])
                       if idx not in mne.pick_types(raw.info, meg=True)])
    empty_raws.append(raw)
empty_raw = mne.concatenate_raws(empty_raws)
empty_raws = []

noise_cov = mne.compute_raw_covariance(empty_raw, method='empirical', 
                                       verbose='warning', rank='info')

data = []
names = []
for path in cli_args.raw:

    fname = os.path.basename(path)
    folder = os.path.dirname(path)

    if fname.startswith('subject_'):
        subjects_dir = cli_args.med_recon
        subject = fname.split('_rest')[0]
        trans = os.path.join(folder, subject + '-trans.fif')
    elif fname.startswith('IC_'):
        subjects_dir = cli_args.ic_recon
        subject = 'IC_' + fname.split('_')[2][1:]
        trans = os.path.join(folder, subject + '-trans.fif')
    else:
        raise Exception('Not implemented')

    if not subject in questionnaire_names:
        continue

    names.append(subject)

    print("Opening: " + str(path) + " (" + str(len(data) + 1) + 
          "/~" + str(len(cli_args.raw)) + ")")

    raw = mne.io.read_raw_fif(path, preload=True, verbose='error')
    raw.filter(*prefilter_band)
    raw.resample(sfreq)
    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names'])
                       if idx not in mne.pick_types(raw.info, meg=True)])


    bem = os.path.join(subjects_dir, subject, 'bem',
                       subject+'-inner_skull-bem-sol.fif')
    src_fname = os.path.join(subjects_dir, subject, 'bem',
                             subject + '-vol-' + vol_spacing + '-src.fif')
    src = mne.source_space.read_source_spaces(src_fname, verbose='warning')

    print("Creating forward solution..")
    fwd = mne.make_forward_solution(
        info=raw.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        verbose='warning')

    print("Creating inverse operator..")
    inv = mne.minimum_norm.make_inverse_operator(
        info=raw.info,
        forward=fwd,
        noise_cov=noise_cov,
        depth=None,
        fixed=False,
        verbose='warning')

    tmin = round(raw.times[-1]*0.60)
    tmax = round(raw.times[-1]*0.90)
    # tmin = round(raw.times[-1]*0.60)
    # tmax = round(raw.times[-1]*0.65)

    raw.crop(tmin, tmax)

    stc = mne.minimum_norm.apply_inverse_raw(raw, inv, 0.1, verbose='warning')
    vertices = stc.vertices[0]

    start = time.time()
    filt_data = mne.filter.filter_data(stc.data, sfreq, l_freq=band[0], 
                                       h_freq=band[1], verbose='warning')
    print("filter_data took: " + str(time.time() - start))

    start = time.time()
    env = np.abs(fast_hilbert(filt_data)) 
    print("fast_hilbert took: " + str(time.time() - start))

    start = time.time()
    if decimation_factor > 1:
        decim_env = np.array([decimate(row, decimation_factor) for row in env])
        data.append(decim_env)
    else:
        data.append(env)

    print("Decimation took: " + str(time.time() - start))

common_names = sorted(set(questionnaire_names).intersection(set(names)))

n_subjects = len(common_names)

X = []
Y = []
for name in common_names:
    data_idx = names.index(name)
    quest_idx = questionnaire_names.index(name)

    X_elem = []
    # if name.startswith('subject'):
    #     X_elem.append(0)
    # else:
    #     X_elem.append(1)

    # Gender
    X_elem.append(float(questionnaire[quest_idx][6]))
    # BIS
    X_elem.append(float(questionnaire[quest_idx][0]))
    # BAS
    X_elem.append(float(questionnaire[quest_idx][1]))
    # BAI
    X_elem.append(float(questionnaire[quest_idx][2]))
    # BDI
    X_elem.append(float(questionnaire[quest_idx][3]))
    # Height
    X_elem.append(float(questionnaire[quest_idx][4]))
    # Weight
    X_elem.append(float(questionnaire[quest_idx][5]))

    X.append(X_elem)
    Y.append(data[data_idx].T)

# standardize 
X = np.array(X)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

print("Covariates: ")
print(X)

ica = HPICA(n_components=n_components,
            random_state=random_state,
            n_iter=3,
            n_gaussians=2,
            init_values={},
            algorithm='subspace')

ica.fit(Y, X)

ica_mixing = ica.mixing
pca_means = ica.wh_mean
pca_whitening = ica.wh_matrix
subject_time_courses = []
for subject_idx in range(n_subjects):
    mixing = pca_whitening[subject_idx].T @ ica_mixing[subject_idx]
    subject_time_courses.append(mixing.T)

subject_spatial_maps = [sub_data.T for sub_data in ica.sources]
common_spatial_maps = np.mean(subject_spatial_maps, axis=0)

if not os.path.exists(cli_args.save_path):
    os.makedirs(cli_args.save_path)

figs = ica.plot_evolution()
for fig_idx, fig in enumerate(figs):
    save_fig(fig, os.path.join(cli_args.save_path, 'evolution', str(fig_idx+1).zfill(2) + '.png'), 200)

beta, se = ica.infer(Y, X)

print("\nBETA SUMMARY")
# Print summaries of beta coefficients
for source_idx in range(beta.shape[2]):
    for covariate_idx in range(beta.shape[1]):
        print("Source " + str(source_idx+1) + ", covariate " + str(covariate_idx+1))

        spatial_beta = beta[:, covariate_idx, source_idx]
        spatial_se = se[:, covariate_idx, source_idx]

        min_idx = np.argmin(spatial_beta)
        val = spatial_beta[min_idx]
        lower = val - 1.96 * spatial_se[min_idx]
        upper = val + 1.96 * spatial_se[min_idx]
        z = val / spatial_se[min_idx]
        pvalue = (1 - scipy.stats.norm.cdf(np.abs(val / spatial_se[min_idx]), 0, 1)) * 2
        print("Minimum {0}, lower {1}, upper {2}, pvalue {3}.".format(
            val, lower, upper, pvalue
        ))

        max_idx = np.argmax(spatial_beta)
        val = spatial_beta[max_idx]
        lower = val - 1.96 * spatial_se[max_idx]
        upper = val + 1.96 * spatial_se[max_idx]
        z = val / spatial_se[max_idx]
        pvalue = (1 - scipy.stats.norm.cdf(np.abs(val / spatial_se[max_idx]), 0, 1)) * 2
        print("Maximum {0}, lower {1}, upper {2}, pvalue {3}.".format(
            val, lower, upper, pvalue
        ))


# Plot maps from beta coefficients
for source_idx in range(beta.shape[2]):
    for covariate_idx in range(beta.shape[1]):
        fig, axes = plt.subplots(3)

        fig.suptitle('Inference of ' + str(source_idx+1) + '. source for hierarhical TICA')
        axes[0].set_title('Covariate ' + str(covariate_idx+1))

        y = beta[:, covariate_idx, source_idx]
        y1 = y + 2.58*se[:, covariate_idx, source_idx]
        y2 = y - 2.58*se[:, covariate_idx, source_idx]

        vmax = np.max([np.abs(y), np.abs(y1), np.abs(y)])

        plot_vol_stc_brainmap(y, vertices, vol_spacing, subjects_dir, axes[1], cap=0.0, vmax=vmax)
        plot_vol_stc_brainmap(y1, vertices, vol_spacing, subjects_dir, axes[0], cap=0.0, vmax=vmax)
        plot_vol_stc_brainmap(y2, vertices, vol_spacing, subjects_dir, axes[2], cap=0.0, vmax=vmax)
        save_fig(fig, os.path.join(cli_args.save_path, 'inference',
            'source_' + str(source_idx+1).zfill(2) + '_cov_' + str(covariate_idx+1).zfill(2) + '.png'), 200)

    save_fig(fig, os.path.join(cli_args.save_path, 'inference', 
             str(source_idx+1).zfill(2) + '.png'), 200)

def plot_subject_comparison_plot(data, title):
    fig, ax = plt.subplots()

    ax.set_title(title)

    ax.set_ylim(-len(data), 1)
    ax.set_yticks(range(-len(data) + 1, 1))
    ax.set_yticklabels(reversed(names), fontsize=5)
    ax.axvline(0, linewidth=0.2, color='black')
    for sub_idx, sub_data in enumerate(data):
        ax.axhline(-sub_idx, linewidth=0.2, color='black')
        # remove outliers
        mean = np.mean(sub_data)
        std = np.std(sub_data)
        sub_data = [elem for elem in sub_data if 
                    elem <= mean + 3*std and elem >= mean - 3*std]
        ax.scatter(sub_data, len(sub_data)*[-sub_idx], color='grey', s=0.2)

        # plot mean
        ax.scatter([np.mean(sub_data)], [-sub_idx], color='green', s=5.0)

        # plot std
        std = np.std(sub_data)
        ax.plot([np.mean(sub_data) - std, np.mean(sub_data) + std], 2*[-sub_idx], color='orange', linewidth=1.0)

    return fig

print("Plot summaries of time courses")
for source_idx in range(n_components):
    data = [sub[source_idx] for sub in subject_time_courses]
    title = "Time course summaries"
    fig = plot_subject_comparison_plot(data, title)
    save_fig(fig, os.path.join(cli_args.save_path, 'summaries',
                               'time_course_summmaries_' + str(source_idx+1) + '.png'), 200)


print("Plot pair plot of IC's")
pg = sns.pairplot(pd.DataFrame(common_spatial_maps.T))
save_fig(pg.fig, os.path.join(cli_args.save_path, 'common_pairplot.png'), 100)

print("Plot common spatial maps")
for comp_idx, spatial_map in enumerate(common_spatial_maps):
    fig, ax = plt.subplots()
    plot_vol_stc_brainmap(spatial_map, vertices, vol_spacing, subjects_dir, ax, cap=0.0)
    save_fig(fig, os.path.join(cli_args.save_path, 'common_spatial_maps',
        'map_' + str(comp_idx+1).zfill(2) + '.png'), 100)

print("Plot individual spatial maps")
for subject_idx, spatial_maps in enumerate(subject_spatial_maps):
    for comp_idx, spatial_map in enumerate(spatial_maps):
        fig, ax = plt.subplots()
        plot_vol_stc_brainmap(spatial_map, vertices, vol_spacing, subjects_dir, ax, cap=0.0)
        save_fig(fig, os.path.join(cli_args.save_path, 'subject_spatial_maps', 
           'map_' + str(comp_idx+1).zfill(2) + '_' + names[subject_idx] + '.png'), 100)

