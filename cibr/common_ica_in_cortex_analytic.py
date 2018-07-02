PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('font', size=3)
    matplotlib.use('Agg')

import sys
import argparse
import os
import time

import multiprocessing

import pyface.qt

import mne
import numpy as np
import matplotlib.pyplot as plt 
import sklearn

from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm

from sklearn.base import BaseEstimator
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
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


def create_stc(raw, trans, subject, empty_raw, subjects_dir):
    """
    """
    # mri = os.path.join(subjects_dir, 'fsaverage', 'mri',
    #                    'T1.mgz')

    bem = os.path.join(subjects_dir, subject, 'bem',
                       subject+'-inner_skull-bem-sol.fif')

    # create volumetric source space
    # src = mne.setup_volume_source_space(
    #     subject=subject,
    #     bem=bem,
    #     mri=mri,
    #     subjects_dir=subjects_dir)

    # create surface-based source space
    print "Creating source space.."
    src = mne.setup_source_space(
        subject=subject,
        spacing='ico3',
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
        depth=0.7,
        fixed=False,
        limit_depth_chs=False,
        verbose='warning')

    print "Applying inverse operator.."
    stc = mne.minimum_norm.apply_inverse_raw(
        raw=raw,
        inverse_operator=inv,
        lambda2=0.1,
        method='MNE',
        label=None,
        pick_ori=None,
        verbose='warning')

    # morph to fsaverage
    # stc = mne.morph_data(
    #     subject_from=subject, 
    #     subject_to='fsaverage', 
    #     stc_from=stc, 
    #     grade=4, 
    #     subjects_dir=subjects_dir)

    return stc


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

        return self.results_.predict(X)


def do_common_logistic_regression(data, subject_data, sfreq, component_idxs=None):
    if component_idxs is None:
        component_idxs = range(data.shape[0])

    X, y = [], []
    epoch_length = 2.0
    logit = LogisticRegressionWrapper()

    for subject in subject_data:
        print "Extracting features for subject: " + subject['name']

        start = subject['start']
        heart_interval = subject['intervals']['heart']
        note_interval = subject['intervals']['note']

        # get normalization factor as a mean of heart and note intervals
        # to get subjects comparable

        heart_start = int((start + heart_interval[0]) * sfreq)
        heart_end = int((start + heart_interval[1]) * sfreq)

        note_start = int((start + note_interval[0]) * sfreq)
        note_end = int((start + note_interval[1]) * sfreq)

        heart_mean = np.mean(data[component_idxs, heart_start:heart_end], axis=1)
        note_mean = np.mean(data[component_idxs, note_start:note_end], axis=1)

        # normalization = np.array([1.0]*len(component_idxs))
        normalization = (heart_mean + note_mean) / 2.0

        # hearts
        current_time = 0
        while current_time + heart_interval[0] < heart_interval[1] - epoch_length:

            start_idx = int((start + heart_interval[0] + current_time) * sfreq)
            end_idx = int((start + heart_interval[0] + current_time + epoch_length) * sfreq)

            feature = np.mean(
                data[component_idxs, start_idx:end_idx],
                axis=1) / normalization[component_idxs]

            X.append(feature)
            y.append(1)

            current_time += epoch_length

        # notes
        current_time = 0
        while current_time + note_interval[0] < note_interval[1] - epoch_length:

            start_idx = int((start + note_interval[0] + current_time) * sfreq)
            end_idx = int((start + note_interval[0] + current_time + epoch_length) * sfreq)

            feature = np.mean(
                data[component_idxs, start_idx:end_idx],
                axis=1) / normalization[component_idxs]

            X.append(feature)
            y.append(0)

            current_time += epoch_length

    print "Classifying"

    X, y = shuffle(X, y)

    X = scale(X)

    predicted = cross_val_predict(logit, X, y, cv=5) > 0.5

    print "Mean accuracy: " + str(metrics.accuracy_score(y, predicted))

    print "Confusion matrix: "
    print metrics.confusion_matrix(y, predicted)

    results = logit.fit(X, y)
    print results.summary()

    # return component idxs of commonly meaningful components
    return results.pvalues[1:].argsort()[:10]


def do_logistic_regression(data, subject_data, sfreq, component_idxs=None):
    if component_idxs is None:
        component_idxs = range(data.shape[0])

    for subject in subject_data:
        print "Logistic regression for subject: " + subject['name']

        start = subject['start']
        heart_interval = subject['intervals']['heart']
        note_interval = subject['intervals']['note']
        logit = LogisticRegressionWrapper()

        epoch_length = 2.0
        X, y = [], []
        
        # hearts
        current_time = 0
        while current_time + heart_interval[0] < heart_interval[1] - epoch_length:

            start_idx = int((start + heart_interval[0] + current_time) * sfreq)
            end_idx = int((start + heart_interval[0] + current_time + epoch_length) * sfreq)

            feature = np.mean(
                data[component_idxs, start_idx:end_idx],
                axis=1)

            X.append(feature)
            y.append(1)

            current_time += epoch_length

        # notes
        current_time = 0
        while current_time + note_interval[0] < note_interval[1] - epoch_length:

            start_idx = int((start + note_interval[0] + current_time) * sfreq)
            end_idx = int((start + note_interval[0] + current_time + epoch_length) * sfreq)

            feature = np.mean(
                data[component_idxs, start_idx:end_idx],
                axis=1)

            X.append(feature)
            y.append(0)

            current_time += epoch_length

        print "Classifying"

        X, y = shuffle(X, y)

        X = scale(X)

        predicted = cross_val_predict(logit, X, y, cv=5) > 0.5

        print "Mean accuracy: " + str(metrics.accuracy_score(y, predicted))

        print "Confusion matrix: "
        print metrics.confusion_matrix(y, predicted)

        results = logit.fit(X, y)
        print results.summary()


def plot_time_series(save_path, data, page, sfreq, subject_data, component_idxs=None):
    if component_idxs is None:
        component_idxs = range(data.shape[0])

    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    fig_ = plt.figure()
    for i, idx in enumerate(component_idxs):

        series = data[idx]
        axes = fig_.add_subplot(page, (len(component_idxs) - 1) / page + 1, i+1)

        for subject in subject_data:
            intervals = subject['intervals']
            start = subject['start']

            axes.axvspan(
                start + intervals['heart'][0], 
                start + intervals['heart'][1], 
                color='r', alpha=0.15, lw=0.5)

            axes.axvspan(
                start + intervals['note'][0], 
                start + intervals['note'][1], 
                color='b', alpha=0.15, lw=0.5)

        axes.plot(np.array(range(len(series))) / sfreq, series)
        axes.set_xlabel = 'Sample'
        axes.set_ylabel = 'Power'

    # save plotted spectra
    if save_path:
        fig_.savefig(os.path.join(save_path, 'series.png'), dpi=620)


def plot_stc_brainmaps(save_path, mixing, mean, vertices, page, component_idxs=None):
    if component_idxs is None:
        component_idxs = range(mixing.shape[1]) 

    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, idx in enumerate(component_idxs):

        fig_ = plt.figure()

        stc_data = np.abs(mixing[:, idx] + mean)
        stc = mne.source_estimate.SourceEstimate(
            stc_data[:, np.newaxis],
            vertices,
            tstep=0.1,
            tmin=0,
            subject='fsaverage')

        fmin = np.percentile(stc.data, 70)
        fmid = np.percentile(stc.data, 95)
        fmax = np.percentile(stc.data, 99)

        brain = stc.plot(hemi='split', views=['med', 'lat', 'dor'], smoothing_steps=30,
                         surface='inflated',
                         clim={'kind': 'value', 'lims': [fmin, fmid, fmax]},
                         )

        if save_path:

            brain_path = os.path.join(save_path, 'brains')
            if not os.path.exists(brain_path):
                os.makedirs(brain_path)

            path = os.path.join(brain_path,
                'comp_' + str(i+1).zfill(2) + '.png')

            brain.save_image(path)


def extract_intervals(events, sfreq, first_samp):
    """ interoseption intervals """
    intervals = {}

    print events

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
        intervals['heart'] = ((heart_start - first_samp) / sfreq,
                              (heart_end - first_samp) / sfreq)
        intervals['note'] = ((note_start - first_samp) / sfreq,
                             (note_end - first_samp) / sfreq)
    except:
        import traceback; traceback.print_exc()
        raise Exception('Something wrong with the triggers')

    return intervals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raws', nargs='+')
    parser.add_argument('--empty')
    parser.add_argument('--save-path')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    n_components = 24
    page = 8
    sampling_rate = 50

    # process similarly to input data 
    empty_raw = load_raw(cli_args.empty)
    empty_raw.resample(sampling_rate)
    empty_raw, _ = preprocess(empty_raw, filter_=(7, 14))

    empty_raw._data = (
        (empty_raw._data - np.mean(empty_raw._data)) / 
        np.std(empty_raw._data))

    current_time = 0
    subject_data = []
    for path_idx, path in enumerate(cli_args.raws):
        folder = os.path.dirname(path)
        fname = os.path.basename(path)
        subject = fname.split('.fif')[0]
        trans = os.path.join(folder, subject + '-trans.fif')

        print "Handling ", path

        raw = load_raw(path)
        raw.resample(sampling_rate)
        raw, events = preprocess(raw, filter_=(7, 14), min_duration=1)

        intervals = extract_intervals(
            events, 
            raw.info['sfreq'], 
            raw.first_samp)

        raw._data = (raw._data - np.mean(raw._data)) / np.std(raw._data)

        stc = create_stc(
            raw=raw, 
            trans=trans, 
            subject=subject, 
            empty_raw=empty_raw, 
            subjects_dir=subjects_dir) 

        # stc.data = (stc.data - np.mean(stc.data)) / np.std(stc.data)

        subject_item = {}
        subject_item['name'] = subject
        subject_item['stc'] = stc
        subject_item['intervals'] = intervals
        subject_item['start'] = current_time

        current_time += stc.data.shape[1] / raw.info['sfreq']

        subject_data.append(subject_item)

    # decimate to 1hz (iir filter so downsample twice)
    def hilbert_and_decimate(data):
        return decimate(decimate(np.abs(hilbert(data)), 10), sampling_rate/10)

    print "Getting the envelope.."

    stime = time.time()
    envelopes = multiprocessing.Pool(len(subject_data)).map(
        hilbert_and_decimate, 
        [subject['stc'].data for subject in subject_data])
    envelope = np.concatenate(envelopes, axis=1)
    print "Took %f s to hilbert and decimate." % (time.time() - stime)

    vertices = subject_data[0]['stc'].vertices

    print "Envelope shape: " + str(envelope.shape)

    for subject in subject_data:
        del subject['stc']

    ica = FastICA(
        n_components=n_components, 
        algorithm='parallel',
        whiten=True,
        max_iter=10000,
        tol=0.000000001)

    print "Transforming with ICA.."

    data = ica.fit_transform(envelope.T).T

    component_idxs = range(n_components)

    print "Plotting time series.."
    plot_time_series(save_path, data, page, 1.0, subject_data, 
                     component_idxs=component_idxs)

    print "Plotting stc brainmaps.."
    plot_stc_brainmaps(save_path, ica.mixing_, ica.mean_, vertices, page,
                       component_idxs=component_idxs)

    print "Classify common"
    component_idxs = do_common_logistic_regression(data, subject_data, 1.0, 
        component_idxs=component_idxs)

    print "Common regressors: " + str(component_idxs + 1)

    print "Classify individuals using common regressors"
    do_logistic_regression(data, subject_data, 1.0, 
                           component_idxs=component_idxs)

    print "Classify individuals using any regressors"
    do_logistic_regression(data, subject_data, 1.0)


