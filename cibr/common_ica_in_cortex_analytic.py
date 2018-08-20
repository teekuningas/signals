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

from signals.cibr.ica.complex_ica import complex_ica

from signals.cibr.common import load_raw
from signals.cibr.common import preprocess
from signals.cibr.common import get_correlations
from signals.cibr.common import calculate_stft
from signals.cibr.common import arrange_as_matrix
from signals.cibr.common import arrange_as_tensor
from signals.cibr.common import plot_mean_spectra

from icasso import Icasso



def create_vol_stc(raw, trans, subject, empty_raw, subjects_dir):
    """
    """
    
    bem = os.path.join(subjects_dir, subject, 'bem',
                       subject+'-inner_skull-bem-sol.fif')

    src_fname = os.path.join(subjects_dir, subject, 'bem',
                             subject+'-vol-8-src.fif')

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

    print "Creating sensor covariance matrix.."
    noise_cov = mne.compute_raw_covariance(empty_raw, tmin=4, verbose='warning')

    print "Creating inverse operator.."
    inv = mne.minimum_norm.make_inverse_operator(
        info=raw.info,
        forward=fwd,
        noise_cov=noise_cov,
        depth=None,
        fixed=False,
        limit_depth_chs=False,
        verbose='warning')

    print "Applying inverse operator.."
    stc = mne.minimum_norm.apply_inverse_raw(
        raw=raw,
        inverse_operator=inv,
        lambda2=0.1,
        method='dSPM',
        label=None,
        pick_ori=None,
        verbose='warning')

    return stc


def create_stc(raw, trans, subject, empty_raw, spacing, subjects_dir):
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

    print "Creating sensor covariance matrix.."
    noise_cov = mne.compute_raw_covariance(empty_raw, tmin=4, verbose='warning')

    print "Creating inverse operator.."
    inv = mne.minimum_norm.make_inverse_operator(
        info=raw.info,
        forward=fwd,
        noise_cov=noise_cov,
        loose=1.0,
        # depth=0.6,
        depth=None,
        fixed=False,
        limit_depth_chs=False,
        verbose='warning')

    print "Applying inverse operator.."
    stc = mne.minimum_norm.apply_inverse_raw(
        raw=raw,
        inverse_operator=inv,
        lambda2=0.1,
        # method='MNE',
        method='dSPM',
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


def do_common_logistic_regression_subject_means(data, subject_data, sfreq, keys):
    X, y = [], []
    epoch_length = 2.0
    logit = LogisticRegressionWrapper()

    for subject in subject_data:
        print "Extracting features for subject: " + subject['name']

        start = subject['start']

        class_id = 0
        for key, ivals in subject['intervals'].items():
            if key not in keys:
                continue

            print "Class id for " + str(key) + " is " + str(class_id) + "."

            feature_elems = []
            
            for ival in ivals:
                current_time = 0
                while current_time + ival[0] < ival[1] - epoch_length:
                    start_idx = int((start + ival[0] + current_time) * sfreq)
                    end_idx = int((start + ival[0] + current_time + epoch_length) * sfreq)  # noqa

                    feature = np.mean(
                        data[:, start_idx:end_idx],
                        axis=1)

                    feature_elems.append(feature)
 
                    current_time += epoch_length
            if feature_elems:
                X.append(np.mean(feature_elems, axis=0))
                y.append(class_id)

            class_id += 1

    print "Classifying"

    # lets make the group sizes equal
    X, y = resample_to_balance(X, y)

    X, y = shuffle(X, y)

    X = scale(X)

    score, _, pvalue = permutation_test_score(
        logit, X, y, scoring='accuracy', cv=5, n_permutations=500)

    print "Permutation test score: " + str(score)
    print "Permutation test pvalue: " + str(pvalue)

    predicted = cross_val_predict(logit, X, y, cv=5)

    print "Mean accuracy: " + str(metrics.accuracy_score(y, predicted))

    print "Confusion matrix: "
    print metrics.confusion_matrix(y, predicted)

    results = logit.fit(X, y)
    print results.summary()


def do_common_logistic_regression(data, subject_data, sfreq, keys):
    X, y = [], []
    epoch_length = 2.0
    logit = LogisticRegressionWrapper()

    for subject in subject_data:
        print "Extracting features for subject: " + subject['name']

        start = subject['start']

        # means = []
        # for key in subject['intervals']:
        #     if key not in keys:
        #         continue
        #     ivals = subject['intervals'][key]
        #     for ival in ivals:
        #         ival_start = int((start + ival[0]) * sfreq)
        #         ival_end = int((start + ival[1]) * sfreq)
        #         mean = np.mean(data[:, ival_start:ival_end], axis=1)
        #         means.append(mean)
        # 
        # normalization = np.mean(means, axis=0)

        class_id = 0
        for key, ivals in subject['intervals'].items():
            if key not in keys:
                continue

            print "Class id for " + str(key) + " is " + str(class_id) + "."
            
            for ival in ivals:
                current_time = 0
                while current_time + ival[0] < ival[1] - epoch_length:
                    start_idx = int((start + ival[0] + current_time) * sfreq)
                    end_idx = int((start + ival[0] + current_time + epoch_length) * sfreq)  # noqa
                    # feature = np.mean(
                    #     data[:, start_idx:end_idx],
                    #     axis=1) / normalization

                    feature = np.mean(
                        data[:, start_idx:end_idx],
                        axis=1)
 
                    X.append(feature)
                    y.append(class_id)

                    current_time += epoch_length
            class_id += 1

    print "Classifying"

    # lets make the group sizes equal
    X, y = resample_to_balance(X, y)

    X, y = shuffle(X, y)

    X = scale(X)

    score, _, pvalue = permutation_test_score(
        logit, X, y, scoring='accuracy', cv=5, n_permutations=500)

    print "Permutation test score: " + str(score)
    print "Permutation test pvalue: " + str(pvalue)

    predicted = cross_val_predict(logit, X, y, cv=5)

    print "Mean accuracy: " + str(metrics.accuracy_score(y, predicted))

    print "Confusion matrix: "
    print metrics.confusion_matrix(y, predicted)

    results = logit.fit(X, y)
    print results.summary()


def do_logistic_regression(data, subject_data, sfreq, keys, print_summary=True):

    if len(keys) > 2:
        raise Exception('Supports only two-class classification')

    epoch_length = 2.0

    accuracies = []
    pvalues = []
    for subject in subject_data:
        print "Logistic regression for subject: " + subject['name']

        logit = LogisticRegressionWrapper()
        X, y = [], []
        start = subject['start']

        class_id = 0
        for key, ivals in subject['intervals'].items():

            if key not in keys:
                continue

            print "Class id for " + str(key) + " is " + str(class_id) + "."

            for ival in ivals:
                current_time = 0
                while current_time + ival[0] < ival[1] - epoch_length:
                    start_idx = int((start + ival[0] + current_time) * sfreq)
                    end_idx = int((start + ival[0] + current_time + epoch_length) * sfreq)
                    feature = np.mean(
                        data[:, start_idx:end_idx],
                        axis=1)
                    X.append(feature)
                    y.append(class_id)

                    current_time += epoch_length
            class_id += 1
        
        print "Classifying"

        X, y = shuffle(X, y)

        X = scale(X)

        score, _, pvalue = permutation_test_score(
            logit, X, y, scoring='accuracy', cv=5, n_permutations=500)

        accuracies.append(score)
        pvalues.append(pvalue)

        if print_summary:
            print "Mean accuracy: " + str(score)
            print "Confusion matrix: "
            predicted = cross_val_predict(logit, X, y, cv=5)
            print metrics.confusion_matrix(y, predicted)
            results = logit.fit(X, y)
            print results.summary()

    return accuracies, pvalues


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


def plot_vol_stc_brainmaps(save_path, name, brainmaps, vertices, subjects_dir):

    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx in range(brainmaps.shape[0]):

        fig_ = plt.figure()

        stc_data = brainmaps[idx]
        stc = mne.source_estimate.VolSourceEstimate(
            stc_data[:, np.newaxis],
            vertices,
            tstep=0.1,
            tmin=0,
            subject='fsaverage')

        src_fname = os.path.join(subjects_dir, 'fsaverage', 'bem', 
                                 'fsaverage-vol-8-src.fif')
        src = mne.source_space.read_source_spaces(src_fname)

        t1_fname = os.path.join(subjects_dir, 'fsaverage', 'mri', 'T1.mgz')
        t1_img = nib.load(t1_fname)

        nifti = stc.as_volume(src).slicer[:, :, :, 0]

        display = plot_glass_brain(t1_img, figure=fig_)
        display.add_overlay(nifti, alpha=0.75)

        plt.show()

        if save_path:

            brain_path = os.path.join(save_path, 'brains')
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
        # np.abs(mixing[:, idx])
        stc = mne.source_estimate.SourceEstimate(
            stc_data[:, np.newaxis],
            vertices,
            tstep=0.1,
            tmin=0,
            subject='fsaverage')

        # fmin = np.percentile(stc.data, 60)
        # fmid = np.percentile(stc.data, 92)
        # fmax = np.percentile(stc.data, 97)

        # fmin = np.percentile(stc.data, 50)
        # fmid = np.percentile(stc.data, 90)
        # fmax = np.percentile(stc.data, 95)

        fmin = np.percentile(stc.data, 50)
        fmid = np.percentile(stc.data, 95)
        fmax = np.percentile(stc.data, 99)

        brain = stc.plot(hemi='split', views=['med', 'lat', 'dor'], 
                         smoothing_steps=30,
                         surface='inflated',
                         clim={'kind': 'value', 'lims': [fmin, fmid, fmax]},
        )

        if save_path:

            brain_path = os.path.join(save_path, 'brains')
            if not os.path.exists(brain_path):
                os.makedirs(brain_path)

            path = os.path.join(brain_path,
                name + '_comp_' + str(idx+1).zfill(2) + '.png')

            brain.save_image(path)


def save_series(save_path, data, sfreq, subject_data, keys):

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
    parser.add_argument('--empty')
    parser.add_argument('--save-path')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    n_components_before_icasso = 30
    n_components_after_icasso = 24
    sampling_rate_raw = 50.0
    sampling_rate_hilbert = 1.0
    band = (7, 14)

    # process similarly to input data 
    empty_raw = load_raw(cli_args.empty)
    empty_raw.resample(sampling_rate_raw)
    empty_raw, _ = preprocess(empty_raw, filter_=band)

    # empty_raw._data = (
    #     (empty_raw._data - np.mean(empty_raw._data)) / 
    #     np.std(empty_raw._data))

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

        # raw._data = (raw._data - np.mean(raw._data)) / np.std(raw._data)

        # stc = create_stc(
        #     raw=raw, 
        #     trans=trans, 
        #     subject=subject, 
        #     empty_raw=empty_raw, 
        #     spacing=spacing,
        #     subjects_dir=subjects_dir) 

        stc = create_vol_stc(
            raw=raw, 
            trans=trans, 
            subject=subject, 
            empty_raw=empty_raw, 
            subjects_dir=subjects_dir) 

        stc.data = (stc.data - np.mean(stc.data)) / np.std(stc.data)

        subject_item = {}
        subject_item['name'] = subject
        subject_item['stc'] = stc
        subject_item['intervals'] = intervals
        subject_item['start'] = current_time

        current_time += stc.data.shape[1] / raw.info['sfreq']

        subject_data.append(subject_item)

    def hilbert_and_decimate(data):
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

    print "Getting the envelope.."

    stime = time.time()
    # envelopes = multiprocessing.Pool(len(subject_data)).map(
    #     hilbert_and_decimate, 
    #     [subject['stc'].data for subject in subject_data])

    envelopes = []
    for subject in subject_data:
        envelopes.append(hilbert_and_decimate(subject['stc'].data))

    envelope = np.concatenate(envelopes, axis=1)
    print "Took %f s to hilbert and decimate." % (time.time() - stime)

    vertices = subject_data[0]['stc'].vertices

    print "Envelope shape: " + str(envelope.shape)

    for subject in subject_data:
        del subject['stc']

    print "Transforming with ICA.."

    ica_params = {
        'n_components': n_components_before_icasso,
        'algorithm': 'parallel',
        'whiten': True,
        'max_iter': 10000,
        'tol': 0.000000001
    }

    print "Fitting icasso."
    icasso = Icasso(FastICA, ica_params=ica_params, iterations=100,
                    bootstrap=True, vary_init=True)

    def bootstrap_fun(data, generator):
        sample_idxs = generator.choice(range(data.shape[0]), size=data.shape[0])
        return data[sample_idxs, :]

    icasso.fit(envelope.T, fit_params={},
               unmixing_fun=lambda ica: ica.components_,
               bootstrap_fun=bootstrap_fun)

    fig_ = icasso.plot_dendrogram()

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fig_.savefig(os.path.join(save_path, 'dendrogram.png'), dpi=620)

    # icasso.plot_mds(distance=0.75)

    unmixing, scores = icasso.get_centrotype_unmixing(distance=0.75)
    mixing = np.linalg.pinv(unmixing)

    data = np.dot(unmixing, envelope)

    # take only `amount` best
    amount = n_components_after_icasso
    data = data[:amount, :]
    mixing = mixing[:, :amount]

    print "Mixing shape: " + str(mixing.shape)

    # ica = FastICA(
    #     n_components=n_components_before_icasso, 
    #     algorithm='parallel',
    #     whiten=True,
    #     max_iter=10000,
    #     tol=0.000000001)

    # data = ica.fit_transform(envelope.T).T
    # mixing = ica.mixing_
    # ica_mean = ica.mean_

    print "Plotting time series.."
    plot_time_series(save_path, data, page, sampling_rate_hilbert, 
                     subject_data, keys=['ec'])

    print "Plotting vol stc brainmaps from ICA mixing matrix.."
    plot_vol_stc_brainmaps(save_path, 'mixing', np.abs(mixing.T), vertices,
                           subjects_dir)

    # print "Plotting stc brainmaps from ICA mixing matrix.."
    # plot_stc_brainmaps(save_path, 'mixing', np.abs(mixing.T), vertices)

    # print "Saving data.."
    save_series(save_path, data, sampling_rate_hilbert, subject_data,
                keys=['ec'])


    ## Classification
    
    # print "Classify common"

    # do_common_logistic_regression_subject_means(data, subject_data, 
    #                                             sampling_rate_hilbert, 
    #                                             keys=['depression', 'control'])

    # do_common_logistic_regression(data, subject_data, sampling_rate_hilbert, 
    #                               keys=['depression', 'control'])

    # print "Classify individuals using any regressors"
    do_logistic_regression(data, subject_data, sampling_rate_hilbert,
                           keys=['plan', 'anx'])

    # print "Test every regressor for every subject"
    # for idx in range(data.shape[0]):
    #     print "Regression for component " + str(idx+1)

    #     do_common_logistic_regression_subject_means(
    #         data[:idx+1][idx:], subject_data, sampling_rate_hilbert, 
    #         keys=['depression', 'control'])

    # raise Exception('Kissa')

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

