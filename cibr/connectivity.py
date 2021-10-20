import argparse
import pickle
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

use_hpica = True

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

# Try reading pickled data.
pickle_fname = os.path.join(cli_args.save_path, 'pickled_data.pickle')

found_pickled = False
if os.path.exists(pickle_fname):
    with open(pickle_fname, 'rb') as f:
        pickle_dict = pickle.load(f)

        common_spatial_maps = pickle_dict['common_spatial_maps']
        subject_spatial_maps = pickle_dict['subject_spatial_maps']
        subject_time_courses = pickle_dict['subject_time_courses']
        vertices = pickle_dict['vertices']
        names = pickle_dict['names']

        # sanity checks
        if len(subject_time_courses) != len(cli_args.raw):
            print("Found pickle file but the amount of subjects does not match.")
            pass
        else:
            found_pickled = True

if not found_pickled:
    print("Could not read a pickle file. Starting from beginning.")

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

        print("Opening: " + str(path) + " (" + str(len(data) + 1) + 
              "/" + str(len(cli_args.raw)) + ")")
        raw = mne.io.read_raw_fif(path, preload=True, verbose='error')
        raw.filter(*prefilter_band)
        raw.resample(sfreq)
        raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names'])
                           if idx not in mne.pick_types(raw.info, meg=True)])

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

        names.append(subject)

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


    if use_hpica:
        from hpica import compute_hpica

        Y = [sub_data.T for sub_data in data]
        results = compute_hpica(Y, n_components=n_components, random_state=random_state,
                                n_iter=10, n_gaussians=3, initial_guess='ica', algorithm='subspace')
        ica_mixing = results[0]
        pca_means = results[6]
        pca_whitening = results[7]

        subject_spatial_maps = []
        subject_time_courses = []
        for subject_idx in range(len(data)):
            demeaned = Y[subject_idx] - pca_means[subject_idx][:, np.newaxis]
            mixing = pca_whitening[subject_idx].T @ ica_mixing[subject_idx]
            unmixing = mixing.T
            sources = (unmixing @ demeaned)

            subject_spatial_maps.append(sources)
            subject_time_courses.append(unmixing)

        common_spatial_maps = np.mean(subject_spatial_maps, axis=0)

    else:
        whitened_data = []
        whitenings = []
        means = []
        for subject_data in data:
            pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
            whitened = pca.fit_transform(subject_data).T
            whitening = ((pca.components_  / pca.singular_values_[:, np.newaxis]) * 
                         np.sqrt(subject_data.shape[0]))
            whitened_data.append(whitened)
            whitenings.append(whitening)
            means.append(pca.mean_)

        concatenated_data = np.concatenate(whitened_data, axis=0)

        pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
        whitened = pca.fit_transform(concatenated_data.T).T
        whitening = ((pca.components_  / pca.singular_values_[:, np.newaxis]) * 
                     np.sqrt(concatenated_data.shape[1]))

        ica = FastICA(whiten=False, random_state=random_state)
        sources = ica.fit_transform(whitened.T).T
        unmixing = ica.components_ @ whitening

        subject_spatial_maps = []
        subject_time_courses = []
        for subject_idx in range(len(data)):
            subject_unmixing = unmixing[:, subject_idx*n_components:(subject_idx+1)*n_components]

            subject_data = data[subject_idx].T - np.mean(data[subject_idx], axis=0)[:, np.newaxis]

            subject_time_courses.append(subject_unmixing @ whitenings[subject_idx])
            subject_spatial_maps.append(subject_unmixing @ whitenings[subject_idx] @ subject_data)

        # common_spatial_maps = np.mean(subject_spatial_maps, axis=0)
        common_spatial_maps = sources

        # save to pickle file
        pickle_dict = {
            'subject_spatial_maps': subject_spatial_maps,
            'subject_time_courses': subject_time_courses,
            'common_spatial_maps': common_spatial_maps,
            'names': names,
            'vertices': vertices,
        }
        if not os.path.exists(cli_args.save_path):
            os.makedirs(cli_args.save_path)
        with open(pickle_fname, 'wb') as f:
            pickle.dump(pickle_dict, f)

common_names = sorted(set(questionnaire_names).intersection(set(names)))

print("Analyse relationship between components and behavioral variables")
do_stats = True
for source_idx in range(n_components):
    if not do_stats:
        break
    print("")
    print("Handling component " + str(source_idx+1))

    do_regression = True
    if do_regression:
        print("")
        print("Running regression analysis.")

        Y = []
        X = []
        for name in common_names:
            data_idx = names.index(name)
            quest_idx = questionnaire_names.index(name)

            X.append(subject_spatial_maps[data_idx][source_idx])
            Y.append(np.array(questionnaire[quest_idx], dtype=np.float))

        reg_pca = PCA(n_components=5, whiten=True, random_state=random_state)
        reg_X = reg_pca.fit_transform(np.array(X))
        reg_whitening = ((reg_pca.components_  / reg_pca.singular_values_[:, np.newaxis]) * 
                         np.sqrt(np.array(X).shape[1]))
        print("Explained variance: ")
        print(reg_pca.explained_variance_ratio_)

        for comp_idx, spatial_map in enumerate(reg_whitening):
            fig, ax = plt.subplots()
            plot_vol_stc_brainmap(spatial_map, vertices, vol_spacing, subjects_dir, ax, cap=0.0)
            path = os.path.join(cli_args.save_path, 'reg_pca', 
                'map_' + str(source_idx+1).zfill(2) + '_' + str(comp_idx+1).zfill(2) + '.png')
            save_fig(fig, path, 100)

        # Let's do regression separately for each behavioral variable
        for behav_idx in range(np.array(Y).shape[1]):
            print("")
            print("Handling " + str(questionnaire_header[behav_idx+1])) 
            behav_vals = np.array(Y)[:, behav_idx]
            behav_vals = (behav_vals - np.mean(behav_vals)) / np.std(behav_vals)

            reg_Y = behav_vals
            fig, ax = plt.subplots()
            print(sm.OLS(reg_Y, sm.add_constant(reg_X)).fit().summary())


    print("")
    print("Running permutation tests for independent components..")

    do_perm_tests = True
    quest_data = np.array(questionnaire).astype(float)
    for behav_idx in range(len(quest_data.T)):
        if not do_perm_tests:
            break
        behav = quest_data[:, behav_idx]
        behav_name = questionnaire_header[1 + behav_idx]
        print("Analysing " + str(behav_name))

        behav_filt = []
        data_filt = []
        common_names = sorted(set(questionnaire_names).intersection(set(names)))
        for name in common_names: 
            quest_idx = questionnaire_names.index(name)
            data_idx = names.index(name); 
            behav_filt.append(behav[quest_idx])
            data_filt.append(subject_spatial_maps[data_idx][source_idx])

        behav_filt = np.array(behav_filt)
        data_filt = np.array(data_filt)

        low_idxs, mid_idxs, high_idxs = np.array_split(np.argsort(behav_filt), 3)

        low_data = data_filt[low_idxs]
        high_data = data_filt[high_idxs]
  
        low_behavs = behav_filt[low_idxs]
        high_behavs = behav_filt[high_idxs]

        print("Mean for low behavs: " + str(np.mean(low_behavs)))
        print("Mean for high behavs: " + str(np.mean(high_behavs)))

        threshold = 8
        adjacency = coo_matrix(np.corrcoef(data_filt.T))

        results = mne.stats.cluster_level.permutation_cluster_test(
            X=[low_data, high_data],
            threshold=threshold,
            n_permutations=1024,
            adjacency=adjacency,
            out_type='indices',
            verbose=False)

        n_clusters = len(results[2])

        print("F: " + str(results[0]))
        print("pv: "+ str(results[2]))

        mean_difference = np.mean(high_data, axis=0) - np.mean(low_data, axis=0)

        clusters = []
        for cluster_idx in range(n_clusters):
            pvalue =  results[2][cluster_idx]
            cluster_map = np.array([np.abs(np.random.normal(scale=0.001)) for _ in range(high_data.shape[1])])
            for vert_idx in results[1][cluster_idx][0]:
                cluster_map[vert_idx] = 1 + np.random.normal(scale=0.001)

            clusters.append((pvalue, cluster_map))

        fig, ax = plt.subplots()
        plot_vol_stc_brainmap(mean_difference, vertices, vol_spacing, subjects_dir, ax, cap=0.0)
        path = os.path.join(cli_args.save_path, 'group_comparisons', 'mean_difference_' + 
                            str(source_idx+1).zfill(2) + '_' + behav_name + '.png')
        save_fig(fig, path, 100)

        # plot clusters
        for cluster_idx in range(n_clusters):
            fig, ax = plt.subplots()
            plot_vol_stc_brainmap(clusters[cluster_idx][1], vertices, vol_spacing, 
                                  subjects_dir, ax, cmap='PiYG', cap=0.0)
            path = os.path.join(cli_args.save_path, 'group_comparisons', 'cluster_' + 
                                str(source_idx+1).zfill(2) + '_' + behav_name + 
                                '_' + str(cluster_idx+1).zfill(2) + '.png')
            save_fig(fig, path, 100)



    print("")
    print("Compare time courses..")

    quest_data = np.array(questionnaire).astype(float)
    for behav_idx in range(len(quest_data.T)):
        behav = quest_data[:, behav_idx]
        behav_name = questionnaire_header[1 + behav_idx]
        print("Analysing " + str(behav_name))

        behav_filt = []
        data_filt = []
        common_names = sorted(set(questionnaire_names).intersection(set(names)))
        for name in common_names: 
            quest_idx = questionnaire_names.index(name)
            data_idx = names.index(name); 
            behav_filt.append(behav[quest_idx])
            data_filt.append(subject_time_courses[data_idx][source_idx])

        behav_filt = np.array(behav_filt)
        data_filt = np.array(data_filt)

        low_idxs, mid_idxs, high_idxs = np.array_split(np.argsort(behav_filt), 3)

        low_data = data_filt[low_idxs]
        high_data = data_filt[high_idxs]
  
        low_behavs = behav_filt[low_idxs]
        high_behavs = behav_filt[high_idxs]

        print("")
        print("Mean for low behavs: " + str(np.mean(low_behavs)))
        print("Mean for high behavs: " + str(np.mean(high_behavs)))

        low_means = [np.mean(sub_data) for sub_data in low_data]
        high_means = [np.mean(sub_data) for sub_data in high_data]

        print("")
        print("Mean for low data means: " + str(np.mean(low_means)))
        print("Mean for high data means: " + str(np.mean(high_means)))
        print("Ttest: " + str(scipy.stats.ttest_ind(low_means, high_means)))

        low_vars = [np.var(sub_data) for sub_data in low_data]
        high_vars = [np.var(sub_data) for sub_data in high_data]

        print("")
        print("Mean for low vars: " + str(np.mean(low_vars)))
        print("Mean for high vars: " + str(np.mean(high_vars)))
        print("Ttest: " + str(scipy.stats.ttest_ind(low_vars, high_vars)))


if not os.path.exists(cli_args.save_path):
    os.makedirs(cli_args.save_path)


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
