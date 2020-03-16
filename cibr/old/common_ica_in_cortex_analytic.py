PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('font', size=3)
    matplotlib.use('Agg')

import sys
import csv
import argparse
import os
import time

import multiprocessing

from collections import OrderedDict

import pyface.qt

import nibabel as nib

from nilearn.plotting import plot_glass_brain

import mne
import numpy as np
import matplotlib.pyplot as plt 
import sklearn
import pandas as pd
import seaborn as sns

from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm

from sklearn.base import BaseEstimator
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import permutation_test_score
from sklearn import metrics
from sklearn.preprocessing import scale

from scipy.signal import hilbert
from scipy.signal import decimate
from sklearn.decomposition import FastICA

from signals.cibr.common import load_raw
from signals.cibr.common import preprocess
from signals.cibr.common import calculate_stft
from signals.cibr.common import arrange_as_matrix
from signals.cibr.common import arrange_as_tensor

from icasso import Icasso
from signals.cibr.ica.complex_ica import ComplexICA
from signals.cibr.ica.complex_ica_alt import ComplexICA as ComplexICAAlt


def create_vol_stc(raw, trans, subject, noise_cov, spacing, 
                   mne_method, mne_depth, subjects_dir):
    """
    """
    
    bem = os.path.join(subjects_dir, subject, 'bem',
                       subject+'-inner_skull-bem-sol.fif')

    src_fname = os.path.join(subjects_dir, subject, 'bem',
                             subject + '-vol-' + spacing + '-src.fif')

    src = mne.source_space.read_source_spaces(src_fname)

    print "Creating forward solution.."
    fwd = mne.make_forward_solution(
        info=raw.info, 
        trans=trans, 
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        verbose='warning')

    print "Creating inverse operator.."
    inv = mne.minimum_norm.make_inverse_operator(
        info=raw.info,
        forward=fwd,
        noise_cov=noise_cov,
        depth=mne_depth,
        fixed=False,
        limit_depth_chs=False,
        verbose='warning')

    print "Applying inverse operator.."
    stc = mne.minimum_norm.apply_inverse_raw(
        raw=raw,
        inverse_operator=inv,
        lambda2=0.1,
        method=mne_method,
        label=None,
        pick_ori=None,
        verbose='warning')

    return stc


def create_stc(raw, trans, subject, noise_cov, spacing, 
               mne_method, mne_depth, subjects_dir):
    """
    """

    bem = os.path.join(subjects_dir, subject, 'bem',
                       subject+'-inner_skull-bem-sol.fif')

    # create surface-based source space
    print "Creating source space.."
    src = mne.setup_source_space(
        subject=subject,
        spacing=spacing,
        surface='white',
        subjects_dir=subjects_dir,
        verbose='warning')

    print "Creating forward solution.."
    fwd = mne.make_forward_solution(
        info=raw.info, 
        trans=trans, 
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        verbose='warning')

    print "Creating inverse operator.."
    inv = mne.minimum_norm.make_inverse_operator(
        info=raw.info,
        forward=fwd,
        noise_cov=noise_cov,
        loose=1.0,
        depth=mne_depth,
        fixed=False,
        limit_depth_chs=False,
        verbose='warning')

    print "Applying inverse operator.."
    stc = mne.minimum_norm.apply_inverse_raw(
        raw=raw,
        inverse_operator=inv,
        lambda2=0.1,
        method=mne_method,
        label=None,
        pick_ori=None,
        verbose='warning')

    return stc


def resample_to_balance(X, y):

    X = np.array(X)
    y = np.array(y)

    class_0_idxs = np.where(y == 0)[0]
    class_1_idxs = np.where(y == 1)[0]

    elem_count = np.min([len(class_0_idxs), len(class_1_idxs)])

    class_0_sample = np.random.choice(range(len(class_0_idxs)), 
                                      size=elem_count,
                                      replace=False)
    class_1_sample = np.random.choice(range(len(class_1_idxs)),
                                      size=elem_count,
                                      replace=False)

    X_0 = X[class_0_idxs[class_0_sample]]
    y_0 = len(X_0)*[0]
    X_1 = X[class_1_idxs[class_1_sample]]
    y_1 = len(X_1)*[1]

    return (np.concatenate([X_0, X_1], axis=0), 
            np.concatenate([y_0, y_1], axis=0))


class LogisticRegressionWrapper(BaseEstimator):
    """  """

    def fit(self, X, y):

        X = sm.add_constant(X)

        self.model_ = Logit(y, X)

        self.results_ = self.model_.fit_regularized(
            method='l1',
            disp=False,
        ) 

        return self.results_

    def predict(self, X):
        
        X = sm.add_constant(X)

        return self.results_.predict(X) > 0.5


def prepare_classification(data, sfreq, subject_data, method='fourier'):
    features, labels = [], []

    for subject in subject_data:

        start = subject['start']

        subject_features = []
        subject_labels = []

        for key, ivals in subject['intervals'].items():

            for ival in ivals:
                start_idx = int((start + ival[0]) * sfreq)
                end_idx = int((start + ival[1]) * sfreq)
                ival_features = []
                for current_idx in range(start_idx, end_idx):
                    if method == 'fourier':
                        feature = np.mean(data[:, :, current_idx]**2, axis=1)
                    else: 
                        feature = data[:, current_idx]**2
                    ival_features.append(feature)
                subject_features.append(ival_features)
                subject_labels.append(key)
        features.append(subject_features) 
        labels.append(subject_labels)

    return features, labels


def transform_to_blocks(features, labels, keys, block_size=5):
    feature_blocks = []
    label_blocks = []

    key1_features = [feature for idx, feature in enumerate(features) if labels[idx] == keys[0]]
    key2_features = [feature for idx, feature in enumerate(features) if labels[idx] == keys[1]]
    key1_label = 0
    key2_label = 1

    ival_count = min([len(key1_features), len(key2_features)])

    shortest_ival = min([len(feature) for feature in features])

    for idx in range(ival_count):
        key1_featurelist = key1_features[idx]
        key2_featurelist = key2_features[idx]
        for elem_idx in range(0, shortest_ival-block_size, block_size):
            key1_feats = key1_featurelist[elem_idx:elem_idx+block_size]
            key2_feats = key2_featurelist[elem_idx:elem_idx+block_size]
            # check that this concatenation works
            feature_blocks.append(key1_feats + key2_feats)
            label_blocks.append(
                [key1_label]*len(key1_feats) + [key2_label]*len(key2_feats))

    return feature_blocks, label_blocks
    
def do_logistic_regression_ind(features, labels, keys, bootstrap_iterations):

    logit = LogisticRegressionWrapper()

    if len(keys) > 2:
        raise Exception('Supports only two-class classification')

    feature_blocks, label_blocks = transform_to_blocks(features, labels, keys)
    
    # X = []
    # y = []
    # for idx in range(len(features)):
    #     if labels[idx] == keys[0]:
    #         y.append(0)
    #     elif labels[idx] == keys[1]:
    #         y.append(1)
    #     else:
    #         continue
    #     X.append(features[idx])

    # X, y = resample_to_balance(X, y)
    # X, y = shuffle(X, y)
    # X = scale(X)

    # score, _, pvalue = permutation_test_score(
    #     logit, X, y, scoring='accuracy', cv=5, n_permutations=500)

    print "Bootstrapping regression."
    bootstrap_results = []
    for bootstrap_idx in range(bootstrap_iterations):
        if bootstrap_idx % 100 == 0:
            print "Took " + str(bootstrap_idx) + " bootstrap samples already."

        sample_blocks = np.random.choice(range(len(label_blocks)), size=len(label_blocks), replace=True)
        sample_X = []
        sample_y = []
        for sample_idx in sample_blocks:
            sample_X.extend(feature_blocks[sample_idx])
            sample_y.extend(label_blocks[sample_idx])

        sample_X, sample_y = shuffle(sample_X, sample_y)
        sample_X = scale(sample_X)

        # sample_idxs = np.random.choice(range(len(y)), size=len(y), replace=True)
        # sample_X = X[sample_idxs]
        # sample_y = y[sample_idxs]

        regression_results = logit.fit(sample_X, sample_y)
        bootstrap_results.append(regression_results.params[1:])
    
    print "Boostrap results: "
    params = []
    significance = []
    bootstrap_results = np.array(bootstrap_results)
    for idx in range(bootstrap_results.shape[1]):
        mean = np.mean(bootstrap_results[:, idx])
        lower, upper = np.percentile(bootstrap_results[:, idx], [0.2, 99.8])
        params.append(mean)
        sigf = False if lower < 0 < upper else True
        significance.append(sigf)
        print "Component " + str(idx+1) + ": " + str(mean) + ',' + str(lower) + ', ' + str(upper) + ', ' + str(sigf)

    # print "Mean accuracy: " + str(score)
    # print "P value: " + str(pvalue)
    # print "Confusion matrix: "
    # predicted = cross_val_predict(logit, X, y, cv=5)
    # print metrics.confusion_matrix(y, predicted)

    # print "Fitting model using all individual data."
    # results = logit.fit(X, y)
    # print results.summary()

    # return results.params[1:], results.pvalues[1:]
    return params, significance



def do_logistic_regression_subjects(features, labels, keys, bootstrap_iterations):

    logit = LogisticRegressionWrapper()

    if len(keys) > 2:
        raise Exception('Supports only two-class classification')

    accuracies = []
    all_idxs = range(len(features))
    for idx in all_idxs:
        test_subject_idx = idx
        train_subject_idxs = all_idxs[:idx] + all_idxs[idx+1:]

        # train classifier
        print "Training classifier using all but subject " + str(idx+1)
        train_X = []
        train_y = []
        for subject_idx in train_subject_idxs:
            subject_train_X = []
            subject_train_y = []
            for block_idx in range(len(labels[subject_idx])):
                if labels[subject_idx][block_idx] == keys[0]:
                    subject_train_y.extend([0]*len(features[subject_idx][block_idx]))
                elif labels[subject_idx][block_idx] == keys[1]:
                    subject_train_y.extend([1]*len(features[subject_idx][block_idx]))
                else:
                    continue
                subject_train_X.extend(features[subject_idx][block_idx])

            subject_train_X = scale(subject_train_X)
            train_X.extend(subject_train_X)
            train_y.extend(subject_train_y)

        train_X, train_y = resample_to_balance(train_X, train_y)
        train_X, train_y = shuffle(train_X, train_y)
        train_X = scale(train_X)

        test_X = []
        test_y = []
        for block_idx in range(len(labels[test_subject_idx])):
            if labels[test_subject_idx][block_idx] == keys[0]:
                test_y.extend([0]*len(features[test_subject_idx][block_idx]))
            elif labels[test_subject_idx][block_idx] == keys[1]:
                test_y.extend([1]*len(features[test_subject_idx][block_idx]))
            else:
                continue
            test_X.extend(features[test_subject_idx][block_idx])

        test_X, test_y = resample_to_balance(test_X, test_y)
        test_X, test_y = shuffle(test_X, test_y)
        test_X = scale(test_X)

        logit.fit(train_X, train_y)

        predicted = logit.predict(test_X)
        accuracy = sklearn.metrics.accuracy_score(test_y, predicted)
        accuracies.append(accuracy)

    print "Mean accuracy score: " + str(np.mean(accuracies))
    print "Individual scores: "
    for idx, accuracy in enumerate(accuracies):
       print  "Subject " + str(idx+1) + ": " + str(accuracy)

    print "Find confidence intervals for coefficients using bootstrap"
    feature_blocks, label_blocks = [], []
    for idx in range(len(features)):
        sub_feat_blocks, sub_lab_blocks = transform_to_blocks(features[idx], 
                                                              labels[idx], keys)
        feature_blocks.extend(sub_feat_blocks)
        label_blocks.extend(sub_lab_blocks)
        
    print "Bootstrapping regression."
    bootstrap_results = []
    for bootstrap_idx in range(bootstrap_iterations):
        if bootstrap_idx % 100 == 0:
            print "Took " + str(bootstrap_idx) + " bootstrap samples already."

        sample_blocks = np.random.choice(range(len(label_blocks)), size=len(label_blocks), replace=True)
        sample_X = []
        sample_y = []
        for sample_idx in sample_blocks:
            sample_X.extend(feature_blocks[sample_idx])
            sample_y.extend(label_blocks[sample_idx])

        sample_X, sample_y = shuffle(sample_X, sample_y)
        sample_X = scale(sample_X)

        regression_results = logit.fit(sample_X, sample_y)
        bootstrap_results.append(regression_results.params[1:])
    
    print "Boostrap results: "
    bootstrap_results = np.array(bootstrap_results)
    for idx in range(bootstrap_results.shape[1]):
        mean = np.mean(bootstrap_results[:, idx])
        lower, upper = np.percentile(bootstrap_results[:, idx], [0.2, 99.8])
        sigf = False if lower < 0 < upper else True
        print "Component " + str(idx+1) + ": " + str(mean) + ',' + str(lower) + ', ' + str(upper) + ', ' + str(sigf)


def plot_time_series(save_path, data, page, sfreq, subject_data, keys):

    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    fig_ = plt.figure()
    for comp_idx in range(data.shape[0]):

        series = data[comp_idx]
        axes = fig_.add_subplot(page, (data.shape[0] - 1) / page + 1, 
                                comp_idx+1)

        for subject in subject_data:
            intervals = subject['intervals']
            start = subject['start']

            colors = ['red', 'blue', 'green', 'yellow']

            for idx, (key, ivals) in enumerate(intervals.items()):
                if key not in keys:
                    continue
                
                color = colors[idx]
                for ival in ivals:
                    axes.axvspan(
                        start + ival[0], 
                        start + ival[1], 
                        color=color, alpha=0.15, lw=0.2)

        axes.plot(np.array(range(len(series))) / sfreq, series,
                  linewidth=0.5)
        axes.set_xlabel = 'Sample'
        axes.set_ylabel = 'Power'

    # save plotted spectra
    if save_path:
        fig_.tight_layout()
        fig_.savefig(os.path.join(save_path, 'series.png'), dpi=620)


def plot_vol_stc_brainmaps(save_path, name, brainmaps, vertices, spacing, subjects_dir):

    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx in range(brainmaps.shape[0]):

        fig_ = plt.figure()

        stc_data = (brainmaps[idx] - np.mean(brainmaps[idx])) / np.std(brainmaps[idx])

        stc = mne.source_estimate.VolSourceEstimate(
            stc_data[:, np.newaxis],
            vertices,
            tstep=0.1,
            tmin=0,
            subject='fsaverage')

        src_fname = os.path.join(subjects_dir, 'fsaverage', 'bem', 
                                 'fsaverage-vol-' + spacing + '-src.fif')
        src = mne.source_space.read_source_spaces(src_fname)

        t1_fname = os.path.join(subjects_dir, 'fsaverage', 'mri', 'T1.mgz')
        t1_img = nib.load(t1_fname)

        nifti = stc.as_volume(src).slicer[:, :, :, 0]

        display = plot_glass_brain(t1_img, figure=fig_)
        display.add_overlay(nifti, alpha=0.75)

        plt.show()

        if save_path:

            brain_path = os.path.join(save_path, 'vol_brains')
            if not os.path.exists(brain_path):
                os.makedirs(brain_path)

            path = os.path.join(brain_path,
                name + '_comp_' + str(idx+1).zfill(2) + '.png')

            fig_.savefig(path, dpi=620)


def plot_stc_brainmaps(save_path, name, brainmaps, vertices):

    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx in range(brainmaps.shape[0]):

        fig_ = plt.figure()

        stc_data = brainmaps[idx]
        stc = mne.source_estimate.SourceEstimate(
            stc_data[:, np.newaxis],
            vertices,
            tstep=0.1,
            tmin=0,
            subject='fsaverage')

        fmin = np.percentile(stc.data, 50)
        fmid = np.percentile(stc.data, 95)
        fmax = np.percentile(stc.data, 99)

        brain = stc.plot(hemi='split', views=['med', 'lat', 'dor'], 
                         smoothing_steps=30,
                         surface='inflated',
                         clim={'kind': 'value', 'lims': [fmin, fmid, fmax]},
        )

        if save_path:

            brain_path = os.path.join(save_path, 'surface_brains')
            if not os.path.exists(brain_path):
                os.makedirs(brain_path)

            path = os.path.join(brain_path,
                name + '_comp_' + str(idx+1).zfill(2) + '.png')

            brain.save_image(path)


def get_subject_spectrums(data, sfreq, subject_data):
    spectrums = {}
    for subject in subject_data:
        intervals = subject['intervals']
        start = subject['start']
        name = subject['name']

        if name not in spectrums:
            spectrums[name] = {}

        for ival_idx, (key, ivals) in enumerate(intervals.items()):
            subspectrums = []
            for ival in ivals:
                x1 = int((start + ival[0]) * sfreq)
                x2 = int((start + ival[1]) * sfreq)
                # print str(key), str(name), str(x1), str(x2), str(data.shape)
                subspectrum = data[:, :, x1:x2]
                subspectrums.append(np.mean(subspectrum, axis=-1))
            spectrums[name][key] = np.mean(subspectrums, axis=0)

    return spectrums


def plot_spectrums(save_path, name, key, data, vmin, vmax, freqs, page):

    if save_path:
        save_path = os.path.join(save_path, 'spectrums')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    fig_ = plt.figure()
    for idx in range(data.shape[0]):
        ax = fig_.add_subplot(page, (data.shape[0] - 1) / page + 1, idx+1)
        spectrum = data[idx]
        ax.plot(freqs, spectrum)
        ax.set_ylim(vmin[idx], vmax[idx])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude (AU)')

    if save_path:
        sub_path = os.path.join(save_path, name + '_' + key + '.png')
        fig_.savefig(sub_path, dpi=310)


def plot_boxplot(title, save_path, save_name, coeffs, sigf_mask):
    """
    """

    # create necessary savepath
    if save_path:
        save_path = os.path.join(save_path, 'boxplots')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # create four subplot columns
    fig_, axes = plt.subplots()

    # Mind, plan and anx and for the first subplot
    fig_.suptitle(title)

    outlier_threshold = 4.0

    sns.boxplot(coeffs, 
                ax=axes, 
                orient='h', 
                whis=outlier_threshold, 
                showcaps=False,
                showmeans=True,
                boxprops={'facecolor': 'None'},
                showfliers=False, 
                whiskerprops={'linewidth': 0}
    )

    # detect outliers

    if np.array(coeffs).size > 0:

        quartile_1, quartile_3 = np.percentile(coeffs, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * outlier_threshold)
        upper_bound = quartile_3 + (iqr * outlier_threshold)

        bounded_coeffs = coeffs[(coeffs >= lower_bound) & (coeffs <= upper_bound)]
        bounded_mask = sigf_mask[(coeffs >= lower_bound) & (coeffs <= upper_bound)]

        # prepare for swarmplot
        data = {'coefficient': bounded_coeffs, 'significance': bounded_mask, 'placeholder': [0]*len(bounded_mask)}
        frame = pd.DataFrame(data)

        # sns.swarmplot(bounded_coeffs, 
        #               ax=axes, 
        #               color='red', 
        #               orient='h'
        # )
        sns.swarmplot(x='coefficient', y='placeholder', hue='significance', data=frame, orient='h',
                      ax=axes)

    axes.axvline(0, linestyle='dashed')

    if save_path:
        sub_path = os.path.join(save_path, save_name + '.png')
        fig_.savefig(sub_path, dpi=620)


def save_means(save_path, data, sfreq, subject_data, keys):

    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    header = ['']
    for subject in subject_data:
        name = subject['name']
        intervals = subject['intervals']
        for key in intervals:
            if key not in keys:
                continue
            header.append(name + ' [' + key + ']')

    data_array = []
    for comp_idx in range(data.shape[0]):
        row = ['Component ' + str(comp_idx+1).zfill(2)]
        for subject in subject_data:
            intervals = subject['intervals']
            start = subject['start']
            for ival_idx, (key, ivals) in enumerate(intervals.items()):
                if key not in keys:
                    continue
                submeans = []
                for ival in ivals:
                    x1 = int((start + ival[0]) * sfreq)
                    x2 = int((start + ival[1]) * sfreq)
                    # print str(x1), str(x2), str(data.shape)
                    submeans.append(np.mean(data[comp_idx, x1:x2]))
                row.append(np.mean(submeans))

        data_array.append(row)

    if save_path:
        with open(os.path.join(save_path, 'power.csv'), 'wb') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)
            for row in data_array:
                writer.writerow(row)


def extract_intervals_interoseption(events, sfreq, first_samp):
    """ interoseption intervals """
    intervals = OrderedDict()

    for event in events:
        if event[2] == 1:
            heart_start = event[0] + 2*sfreq
            continue
        if event[2] == 24:
            heart_end = event[0] - 2*sfreq
            continue
        if event[2] == 5:
            note_start = event[0] + 2*sfreq
            continue
        if event[2] == 28:
            note_end = event[0] - 2*sfreq
            continue
    try:
        intervals['heart'] = [((heart_start - first_samp) / sfreq,
                              (heart_end - first_samp) / sfreq)]
        intervals['note'] = [((note_start - first_samp) / sfreq,
                             (note_end - first_samp) / sfreq)]
    except:
        import traceback; traceback.print_exc()
        raise Exception('Something wrong with the triggers')

    return intervals


def extract_intervals_fdmsa_rest(subject_name):
    """ one key for fdmsa """
    intervals = OrderedDict()

    # start at around 20 seconds
    start = 20 

    # end around 20 seconds before the end
    end = 460

    intervals['ec'] = [(start, end)]
    
    if 'FDMSA_D' in subject_name:
        intervals['depression'] = [(start, end)]
        intervals['control'] = []
    else:
        intervals['depression'] = []
        intervals['control'] = [(start, end)]

    return intervals


def extract_intervals_meditation(events, sfreq, first_samp):
    """ meditation intervals """
    intervals = OrderedDict()

    trigger_info = [
        ('mind', 10), 
        ('rest', 11),
        ('plan', 12),
        ('anx', 13),
    ]

    for name, event_id in trigger_info:
        intervals[name] = []

    for idx, event in enumerate(events):
        for name, event_id in trigger_info:
            if event[2] == event_id:
                ival_start = event[0] + 2*sfreq
                try:
                    ival_end = events[idx+1][0] - 2*sfreq
                except:
                    # last trigger (rest)
                    ival_end = event[0] + 120*sfreq - 2*sfreq

                intervals[name].append((
                    (ival_start - first_samp) / sfreq, 
                    (ival_end - first_samp) / sfreq))

    return intervals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raws', nargs='+')
    parser.add_argument('--empty', nargs='+')
    parser.add_argument('--save-path')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    # hilbert_envelope, TFICA, eTFICA
    ica_method = 'hilbert_envelope'
    src_method = 'vol'
    mne_method, mne_depth = 'dSPM', None
    band = (7, 14)
    sampling_rate_raw = 50.0

    use_icasso = False
    n_components_before_icasso = 30
    n_components_after_icasso = 24
    surf_spacing = 'ico3'
    vol_spacing = '10'
    icasso_threshold = 0.75
    icasso_iterations = 2000
    bootstrap_iterations = 2000
    page = 4

    # hilbert ica settings
    sampling_rate_hilbert = 1.0

    # fourier envelope settings
    window_in_seconds = 1.0
    window_in_samples = np.power(2, np.ceil(np.log(
        sampling_rate_raw * window_in_seconds)/np.log(2)))
    # overlap_in_samples = (window_in_samples * 3) / 4
    # sampling_rate_fourier = (sampling_rate_raw/window_in_samples)*4.0
    overlap_in_samples = 0
    sampling_rate_fourier = (sampling_rate_raw/window_in_samples)

    # process similarly to input data 
    empty_paths = cli_args.empty
    empty_raws = []
    for fname in empty_paths:
        raw = load_raw(fname)
        raw.resample(sampling_rate_raw)
        raw, _ = preprocess(raw, filter_=band)
        empty_raws.append(raw)
    empty_raw = mne.concatenate_raws(empty_raws)
    empty_raws = []

    print "Creating sensor covariance matrix.."
    noise_cov = mne.compute_raw_covariance(
        empty_raw, 
        method='empirical')

    current_time = 0
    subject_data = []
    for path_idx, path in enumerate(cli_args.raws):
        folder = os.path.dirname(path)
        fname = os.path.basename(path)

        # get recon subject name from file name
        
        # this is for interoseption
        # subject = fname.split('.fif')[0]

        # this is for meditation
        subject = '_'.join(fname.split('_')[:2])

        # this is for fdmsa
        # subject = '_'.join((fname.split('_')[:1] + fname.split('_')[2:-2]))

        print "Using MRI subject: " + subject

        trans = os.path.join(folder, subject + '-trans.fif')

        print "Handling ", path

        raw = load_raw(path)
        raw.resample(sampling_rate_raw)
        raw, events = preprocess(raw, filter_=band, min_duration=1)

        # intervals = extract_intervals_interoseption(
        #     events, 
        #     raw.info['sfreq'], 
        #     raw.first_samp)

        intervals = extract_intervals_meditation(
            events, 
            raw.info['sfreq'], 
            raw.first_samp)

        # intervals = extract_intervals_fdmsa_rest(subject)

        if src_method == 'surf':
            stc = create_stc(
                raw=raw, 
                trans=trans, 
                subject=subject, 
                noise_cov=noise_cov, 
                spacing=surf_spacing,
                mne_method=mne_method,
                mne_depth=mne_depth,
                subjects_dir=subjects_dir) 
        else:
            stc = create_vol_stc(
                raw=raw, 
                trans=trans, 
                subject=subject, 
                noise_cov=noise_cov, 
                spacing=vol_spacing,
                mne_method=mne_method,
                mne_depth=mne_depth,
                subjects_dir=subjects_dir) 

        stc.data = (stc.data - np.mean(stc.data)) / np.std(stc.data)

        subject_item = {}
        subject_item['name'] = subject
        subject_item['intervals'] = intervals
        subject_item['start'] = current_time

        def prepare_fourier(data, take_abs=True):
            freqs, times, data, _ = calculate_stft(
                data, sampling_rate_raw, window_in_samples, overlap_in_samples, 
                band[0], band[1], row_wise=True)
            if take_abs:
                return np.abs(arrange_as_matrix(data)), data.shape, freqs
            else:
                return arrange_as_matrix(data), data.shape, freqs

        def prepare_hilbert(data):
            # get envelope as abs of analytic signal
            datas = np.array_split(data, 20, axis=0)
            envs = [np.abs(hilbert(arr)) for arr in datas]
            env = np.concatenate(envs, axis=0)
            # decimate first with five
            decimated = decimate(env, 5)
            # and then the rest
            factor = int(((sampling_rate_raw / 5.0) / sampling_rate_hilbert))
            decimated = decimate(decimated, factor)
            return decimated

        if ica_method == 'hilbert_envelope':
            data = prepare_hilbert(stc.data)
        elif ica_method == 'TFICA':
            data, fourier_shape, freqs = prepare_fourier(stc.data, 
                                                         take_abs=False)
            subject_item['shape'] = fourier_shape
        elif ica_method == 'eTFICA':
            data, fourier_shape, freqs = prepare_fourier(stc.data, 
                                                         take_abs=True)
            subject_item['shape'] = fourier_shape

        subject_item['data'] = data
        current_time += len(raw.times) / raw.info['sfreq']

        vertices = stc.vertices

        subject_data.append(subject_item)

    print "Preparing data for ICA.."

    data = np.concatenate([subject['data'] for subject 
                           in subject_data], axis=1)

    if ica_method == 'eTFICA' or ica_method == 'TFICA':
        fourier_shape = (fourier_shape[0], fourier_shape[1],
                         sum([sbj['shape'][2] for sbj in subject_data])) 

    for subject in subject_data:
        del subject['data']

    print "Shape before ICA: " + str(data.shape)

    print "Transforming with ICA.."

    if ica_method == 'hilbert_envelope' or ica_method == 'eTFICA':
        ica_params = {
            'n_components': n_components_after_icasso,
            'algorithm': 'parallel',
            'whiten': True,
            'max_iter': 10000,
            'tol': 0.000000001
        }

        if use_icasso:
            icasso = Icasso(FastICA, ica_params=ica_params, 
                            iterations=icasso_iterations,
                            bootstrap=True, vary_init=True)

            def bootstrap_fun(data, generator):
                if len(subject_data) >=2:
                    subject_idxs = generator.choice(range(len(subject_data)), 
                                                    size=len(subject_data)-1, 
                                                    replace=False)
                else:
                    subject_idxs = range(len(subject_data))

                if ica_method == 'hilbert_envelope':
                    sfreq = sampling_rate_hilbert
                else:
                    sfreq = sampling_rate_fourier * fourier_shape[1]

                print "Bootstrapping icasso."

                sample_idxs = []
                for subject_idx in subject_idxs:
                    subject = subject_data[subject_idx]
                    intervals = subject['intervals']
                    start = subject['start']
                    name = subject['name']
                    print "Using " + name + " in icasso bootstrap."
                    for ival_idx, (key, ivals) in enumerate(intervals.items()):
                        for ival in ivals:
                            x1 = int((start + ival[0]) * sfreq)
                            x2 = int((start + ival[1]) * sfreq)
                            sample_idxs.extend(range(x1, x2))

                # sample_idxs = generator.choice(range(data.shape[0]), size=data.shape[0])

                return data[sample_idxs, :]

            print "Fitting icasso."
            icasso.fit(data.T, fit_params={},
                       unmixing_fun=lambda ica: ica.components_,
                       bootstrap_fun=bootstrap_fun)
        else:
            ica = FastICA(**ica_params)
            print "Fitting ica."
            ica.fit(data.T)

    elif ica_method == 'TFICA':
        ica_params = {
            'n_components': n_components_after_icasso,
            'conveps': 1e-6,
        }
        if use_icasso:
            icasso = Icasso(ComplexICA, ica_params=ica_params,
                            iterations=icasso_iterations,
                            bootstrap=True, vary_init=True)

            def bootstrap_fun(data, generator):
                sample_idxs = generator.choice(range(data.shape[1]), size=data.shape[1])
                return data[:, sample_idxs]

            print "Fitting icasso."
            icasso.fit(data, fit_params={},
                       unmixing_fun=lambda ica: np.dot(ica.unmixing, ica.whitening),
                       bootstrap_fun=bootstrap_fun)
        else:
            # ica = ComplexICA(**ica_params)
            ica = ComplexICAAlt(**ica_params)
            print "Fitting ica.."
            ica.fit(data)

    if use_icasso:
        print "Plotting dendrogram"
        fig_ = icasso.plot_dendrogram()

        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            fig_.savefig(os.path.join(save_path, 'dendrogram.png'), dpi=620)

        # fig_ = icasso.plot_mds(distance=icasso_threshold)
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            fig_.savefig(os.path.join(save_path, 'mds.png'), dpi=620)

        print "Getting centrotypes"
        unmixing, scores = icasso.get_centrotype_unmixing(distance=icasso_threshold)

        print "Cluster scores: "
        print str(scores)
        mixing = np.linalg.pinv(unmixing)

        data = np.dot(unmixing, data)
        
        # take only `amount` best
        amount = n_components_after_icasso
        data = data[:amount, :]
        mixing = mixing[:, :amount]
    else:
        if ica_method == 'TFICA':
            unmixing = np.dot(ica.unmixing, ica.whitening)
        elif ica_method == 'eTFICA' or ica_method == 'hilbert_envelope':
            unmixing = ica.components_

        data = np.dot(unmixing, data)
        mixing = np.linalg.pinv(unmixing)

    print "Mixing shape: " + str(mixing.shape)

    if ica_method == 'eTFICA' or ica_method == 'TFICA':
        print "Arranging as tensor"
        data = arrange_as_tensor(data, fourier_shape)

    if src_method == 'vol':
        print "Plotting vol stc brainmaps from ICA mixing matrix.."
        plot_vol_stc_brainmaps(save_path, 'mixing', np.abs(mixing.T), vertices,
                               vol_spacing, subjects_dir)

    else:
        print "Plotting surf stc brainmaps from ICA mixing matrix.."
        plot_stc_brainmaps(save_path, 'mixing', np.abs(mixing.T), vertices)

    if ica_method == 'hilbert_envelope':
        print "Plotting time series.."
        plot_time_series(save_path, data, page, sampling_rate_hilbert, 
                         subject_data, keys=['mind', 'plan', 'anx'])
    elif ica_method == 'eTFICA':
        spectrums = get_subject_spectrums(data, sampling_rate_fourier,
                                          subject_data)

        # get vmax for all components
        vmax = [0]*data.shape[0]
        vmin = [None]*data.shape[0]
        for subject in spectrums.keys():
            for key, value in spectrums[subject].items():
                for comp_idx in range(value.shape[0]):
                    if np.max(value[comp_idx]**2) > vmax[comp_idx]:
                        vmax[comp_idx] = np.max(value[comp_idx]**2)
                    if vmin[comp_idx] == None or np.min(value[comp_idx]**2) < vmin[comp_idx]:
                        vmin[comp_idx] = np.min(value[comp_idx]**2)

        for idx in range(len(vmax)):
            vmax[idx] = vmax[idx] / 3.0

        print "Plotting spectrum.."
        for subject in spectrums.keys():
            for key, value in spectrums[subject].items():
                plot_spectrums(save_path, subject, key, value**2, vmin, vmax, freqs, page)
        
        for key in spectrums.values()[0].keys():
            spectrum = np.mean([struct[key] for struct in spectrums.values()], axis=0)
            plot_spectrums(save_path, 'average', key, spectrum**2, vmin, vmax, freqs, page)

    elif ica_method == 'TFICA':
        spectrums = get_subject_spectrums(data, sampling_rate_fourier,
                                          subject_data)

        print "Plotting spectrum.."
        for subject in spectrums.keys():
            for key, value in spectrums[subject].items():
                plot_spectrums(save_path, subject, key, np.abs(value)**2, 
                               freqs, page)
        
        for key in spectrums.values()[0].keys():
            spectrum = np.mean([np.abs(struct[key])**2 for struct 
                                in spectrums.values()], axis=0)
            plot_spectrums(save_path, 'average', key, spectrum, freqs, page)

    # print "Saving data.."
    # if ica_method == 'hilbert_envelope':
    #     save_means(save_path, data, sampling_rate_hilbert, subject_data,
    #                 keys=['mind', 'plan'])
    # else:
    #     # think about how to do this!!
    #     # should I do it like before, to get the freq and amplitude
    #     # with corr?
    #     pass

    ## Classification

    if ica_method == 'eTFICA' or ica_method == 'TFICA':
        features, labels = prepare_classification(
            data, sampling_rate_fourier, subject_data, method='fourier')
    else:
        features, labels = prepare_classification(
            data, sampling_rate_hilbert, subject_data, method='hilbert')

    def plot_and_regress(keys):
        print keys[0] + ' vs ' + keys[1] + ' all'
        coefficients, significance = [], []
        for idx in range(len(subject_data)):
            print "Classification for " + subject_data[idx]['name']
            coeffs, sigfs = do_logistic_regression_ind(features[idx], labels[idx],
                                                       keys, bootstrap_iterations)
            coefficients.append(coeffs)
            significance.append(sigfs)

        coefficients = np.array(coefficients)
        significance = np.array(significance)

        # plot boxplots for each component
        for comp_idx in range(coefficients.shape[1]):
            print "Plot boxplot for component " + str(comp_idx+1)
            comp_coeffs = coefficients[:, comp_idx]
            comp_significance = significance[:, comp_idx]

            title = ('Coefficients (all) of ' + keys[0] + '-' + keys[1] + 
                     ' for each subject of component ' + str(comp_idx+1))
            save_name = 'comp_' + keys[0] + keys[1] + '_' + str(comp_idx+1).zfill(2) + '_all'
            plot_boxplot(title,
                         save_path, 
                         save_name,
                         comp_coeffs,
                         comp_significance)

            # significant_coeffs = np.array([coef for idx, coef in enumerate(comp_coeffs)
            #                                if comp_significance[idx]])

            # title = ('Significant oefficients of ' + keys[0] + '-' + keys[1] + 
            #          ' for each subject of component ' + str(comp_idx+1))
            # save_name = 'comp_' + keys[0] + keys[1] + '_' + str(comp_idx+1).zfill(2)
            # plot_boxplot(title, 
            #              save_path, 
            #              save_name, 
            #              significant_coeffs)

    plot_and_regress(keys=['mind', 'anx'])
    plot_and_regress(keys=['anx', 'plan'])
    plot_and_regress(keys=['mind', 'plan'])

    print "Mind vs anx"
    do_logistic_regression_subjects(features, labels,
                                    ['mind', 'anx'],
                                    bootstrap_iterations)

    print "Anx vs plan"
    do_logistic_regression_subjects(features, labels,
                                    ['anx', 'plan'],
                                    bootstrap_iterations)

    print "Mind vs plan"
    do_logistic_regression_subjects(features, labels,
                                    ['mind', 'plan'],
                                    bootstrap_iterations)

    raise Exception('The end.')
    
    # print "Classify common"

    # do_common_logistic_regression_subject_means(data, subject_data, 
    #                                             sampling_rate_hilbert, 
    #                                             keys=['depression', 'control'])

    # do_common_logistic_regression(data, subject_data, sampling_rate_hilbert, 
    #                               keys=['depression', 'control'])

    # print "Classify individuals using any regressors"

    # do_logistic_regression(data, subject_data, sampling_rate_hilbert,
    #                        keys=['mind', 'anx'])

    # print "Test every regressor for every subject"
    # for idx in range(data.shape[0]):
    #     print "Regression for component " + str(idx+1)

    #     do_common_logistic_regression_subject_means(
    #         data[:idx+1][idx:], subject_data, sampling_rate_hilbert, 
    #         keys=['depression', 'control'])


    print "Test every regressor for every subject"
    component_accuracies = []
    component_pvalues = []
    for idx in range(data.shape[0]):
        print "Regression for component " + str(idx+1)
        accuracies, pvalues = do_logistic_regression(
            data[:idx+1][idx:], subject_data, sampling_rate_hilbert,
            keys=['plan', 'anx'],
            print_summary=False)

        component_accuracies.append(accuracies)
        component_pvalues.append(pvalues)

    for comp_idx in range(data.shape[0]):
        print ("Subject accuracies for component " + str(comp_idx+1) + ": " +
               str(['%0.2f' % acc for acc in component_accuracies[comp_idx]]))
        print ("Subject pvalues for component " + str(comp_idx+1) + ": " +
               str(['%0.2f' % pval for pval in component_pvalues[comp_idx]]))
        print ("Mean accuracy for component " + str(comp_idx+1) + " is: " + 
               str(np.mean(component_accuracies[comp_idx])))

        vals = []
        for sub_idx in range(len(component_accuracies[comp_idx])):
            if component_pvalues[comp_idx][sub_idx] <= 0.05:
                vals.append(component_accuracies[comp_idx][sub_idx])
            else:
                vals.append(0.5)

        print ("Non-signicant as 0.5 mean for component " + 
               str(comp_idx+1) + " is: " + str(np.mean(vals)))

        vals = []
        for sub_idx in range(len(component_accuracies[comp_idx])):
            if component_pvalues[comp_idx][sub_idx] <= 0.05:
                vals.append(component_accuracies[comp_idx][sub_idx])

        if vals:
            print ("Mean of only significant subjects for component " + 
                   str(comp_idx+1) + " is: " + str(np.mean(vals)))
            print "And amount of significant subjects is: " + str(len(vals))

    for sub_idx in range(len(subject_data)):
        print ("Component accuracies for subject " + str(sub_idx+1) + ": " + 
               str(['%0.2f' % acc for acc in 
                    np.array(component_accuracies)[:, sub_idx]]))

