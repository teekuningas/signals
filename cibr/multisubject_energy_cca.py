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
import mne
import numpy as np
import matplotlib.pyplot as plt 
import sklearn
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from statsmodels.discrete.discrete_model import Logit
from statsmodels.multivariate.cancorr import CanCorr
from nilearn.plotting import plot_glass_brain
from sklearn.base import BaseEstimator
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import permutation_test_score
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.cross_decomposition import CCA

from scipy.signal import hilbert
from scipy.signal import decimate
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

from icasso import Icasso

from signals.cibr.common import preprocess


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


def plot_time_series(save_path, data, page, sfreq, subject_data, keys, name):

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
                        color=color, alpha=0.10, lw=0.1)

        axes.plot(np.array(range(len(series))) / sfreq, series,
                  linewidth=0.05)
        axes.set_xlabel = 'Sample'
        axes.set_ylabel = 'Power'

    # save plotted spectra
    if save_path:
        fig_.tight_layout()
        fig_.savefig(os.path.join(save_path, name + '.png'), dpi=620)


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


def plot_barcharts(save_path, components, savename, names):
    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    y_pos = np.arange(len(names))
    for idx, component in enumerate(components):
        print "Plotting barchart"

        fig_, ax = plt.subplots()

        ax.bar(y_pos, component, align='center', alpha=0.5)
        ax.set_xticks(y_pos)
        ax.set_xticklabels(names)
        ax.set_ylabel('CCA weights')

        plt.show()

        if save_path:
            weight_path = os.path.join(save_path, 'weights')
            if not os.path.exists(weight_path):
                os.makedirs(weight_path)
            name = savename + '_' + str(idx+1).zfill(2) + '.png'
            
            path = os.path.join(weight_path, name)
            fig_.savefig(path, dpi=620)

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


def gather_background_data_meditation(subject, raw, events, csv_path, tasks):

    data = []
    header = []

    # first create state data from exp sampling

    subject_code = subject.split('subject_')[-1]

    sfreq = raw.info['sfreq']
    first_samp = raw.first_samp

    trigger_info = [
        ('mind', 10), 
        ('rest', 11),
        ('plan', 12),
        ('anx', 13),
    ]

    ivals = []
    ival_names = []
    for idx, event in enumerate(events):
        for name, event_id in trigger_info:
            # state data only for nonrest tasks
            if name == 'rest':
                continue
            if event[2] == event_id:
                # ival_start = event[0] + 2*sfreq
                ival_start = event[0]
                try:
                    # ival_end = events[idx+1][0] - 2*sfreq
                    ival_end = events[idx+1][0]
                except:
                    # last trigger (rest)
                    # ival_end = event[0] + 120*sfreq - 2*sfreq
                    ival_end = event[0] + 120*sfreq

                ivals.append((ival_start - first_samp,
                              ival_end - first_samp))
                ival_names.append(name)

    def create_state_data(fname):
        data_row = np.array([None]*len(raw.times))
        with open(os.path.join(csv_path, fname), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', skipinitialspace=True)
            for csv_row in reader:
                if csv_row[0].zfill(3) != subject_code:
                    continue

                if len(ivals)/2 != len(csv_row[1:]):
                    print str(len(ivals)), str(len(csv_row))
                    import pdb; pdb.set_trace()
                    raise Exception('Amount of tasks does not match in csv and raw')

                task_values = {}
                for ival_idx, ival in enumerate(ivals):
                    ival_name = ival_names[ival_idx]
                    if ival_name not in task_values:
                        task_values[ival_name] = []
                    val = float(csv_row[1:][ival_idx/2])
                    task_values[ival_name].append(val)

                # values = [float(val) for val in csv_row[1:]]
                # mean_, std_ = np.mean(values), np.std(values)
                # import pdb; pdb.set_trace()

                for ival_idx, ival in enumerate(ivals):
                    if ival_names[ival_idx] not in tasks:
                        continue
                    mean_ = np.mean(task_values[ival_names[ival_idx]])
                    std_ = np.std(task_values[ival_names[ival_idx]])
                    val = (float(csv_row[1:][ival_idx/2]) - mean_) / std_
                    data_row[int(ival[0]):int(ival[1])] = val

        # interpolate Nones
        data_row[data_row == None] = np.nan
        data_row = pd.DataFrame(data_row.astype(np.float)).interpolate(
            limit_direction='both').values[:, 0]

        return data_row

    data.append(create_state_data('feeling.csv'))
    data.append(create_state_data('focused.csv'))
    header.append('feeling')
    header.append('focused')

    data_row = np.array([None]*len(raw.times))

    for ival_idx, ival in enumerate(ivals):
        if ival_names[ival_idx] == tasks[0]:
            val = 0.0
        elif ival_names[ival_idx] == tasks[1]:
            val = 1.0
        else:
            continue
        data_row[int(ival[0]):int(ival[1])] = val

    # interpolate Nones
    data_row[data_row == None] = np.nan
    data_row = pd.DataFrame(data_row.astype(np.float)).interpolate(
        limit_direction='both').values[:, 0]
    # data_row[data_row == None] = 0.5

    data.append(data_row)
    header.append('Task')

    # then add trait data
    with open(os.path.join(csv_path, 'tests.csv'), 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', skipinitialspace=True)
        csv_header = next(reader)
        for csv_row in reader:
            if csv_row[0].zfill(3) != subject_code:
                continue

            for item_idx, item in enumerate(csv_row[1:]):
                # allowed = []
                # allowed = ['BDI', 'BAI', 'BIS', 'BasDrive', 'BasRR', 'BasFS',
                #            'MedLength', 'MedFreq', 'MedExp']
                allowed = ['BDI', 'BAI', 'BIS', 'MedExp']
                if csv_header[1:][item_idx] not in allowed:
                    continue
                data_row = np.array([float(item)]*len(raw.times))
                data.append(data_row)
                header.append(csv_header[1:][item_idx])

    # data.append(data[2]*data[-1])
    # header.append('Task*MedExp')
    # data.append(data[2]*data[-2])
    # header.append('Task*BIS')
    # data.append(data[2]*data[-3])
    # header.append('Task*BAI')
    # data.append(data[2]*data[-4])
    # header.append('Task*BDI')

    return header, np.array(data)

def extract_intervals_meditation(events, sfreq, first_samp, tasks):
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
            if name not in tasks:
                continue
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
    parser.add_argument('--csv-path')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    src_method = 'vol'
    mne_method, mne_depth = 'dSPM', None
    band = (7, 14)
    sampling_rate_raw = 100.0
    tasks = ['mind', 'plan']
    
    surf_spacing = 'ico3'
    vol_spacing = '10'
    bootstrap_iterations = 2000
    page = 4

    # hilbert ica settings
    sampling_rate_hilbert = 1.0

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

        raw = mne.io.Raw(path, preload=True)
        raw.resample(sampling_rate_raw)
        raw, events = preprocess(raw, filter_=band, min_duration=1)

        # intervals = extract_intervals_interoseption(
        #     events, 
        #     raw.info['sfreq'], 
        #     raw.first_samp)

        intervals = extract_intervals_meditation(
            events, 
            raw.info['sfreq'], 
            raw.first_samp,
            tasks)

        print "Gathering background data"
        bg_header, bg_data = gather_background_data_meditation(
            subject, raw, events, cli_args.csv_path,
            tasks)

        # intervals = extract_intervals_fdmsa_rest(subject)


        subject_item = {}
        subject_item['name'] = subject
        subject_item['intervals'] = intervals
        subject_item['bg_header'] = bg_header
        subject_item['start'] = current_time

        def prepare_hilbert(data):
            # get envelope as abs of analytic signal
            datas = np.array_split(data, 20, axis=0)
            # envs = [np.abs(hilbert(arr)) for arr in datas]
            envs = [arr**2 for arr in datas]
            env = np.concatenate(envs, axis=0)
            # decimate first with five
            decimated = decimate(env, 5)
            # and then the rest
            factor = int(((sampling_rate_raw / 5.0) / sampling_rate_hilbert))
            decimated = decimate(decimated, factor)
            return decimated

        decim_factor = int(((sampling_rate_raw / 5.0) / sampling_rate_hilbert))
        bg_data = decimate(decimate(bg_data, 5), decim_factor)

        print "Prepare using hilbert"
        data = prepare_hilbert(raw._data)

        subject_item['data'] = data
        subject_item['bg_data'] = bg_data

        current_time += len(raw.times) / raw.info['sfreq']

        subject_data.append(subject_item)

    print "Preparing data for ICA.."

    # then concatenate them datawise followed by subjectwise concat
    # then look what happens with ICA, and decide whether to use ICA twice
    # also decide whether to normalize the state data


    # concatenate over subjects
    brain_data = np.concatenate([subject['data'] for subject 
                                 in subject_data], axis=1)
    bg_data = np.concatenate([subject['bg_data'] for subject
                              in subject_data], axis=1)

    for subject in subject_data:
        del subject['data']
        del subject['bg_data']

    areas = ['Left-temporal', 'Right-temporal', 'Left-occipital', 'Right-occipital']
    brain_data_wh = []
    for area in areas:
        selected_ch_names = mne.utils._clean_names(
            mne.read_selection(area),
            remove_whitespace=True)
        ch_idxs = [ch_idx for ch_idx, ch_name in enumerate(raw.info['ch_names']) if
                   ch_name in selected_ch_names]
        component = np.mean(brain_data[ch_idxs], axis=0)
        brain_data_wh.append(component)
    brain_data_wh = np.array(brain_data_wh)

    print "Plotting time series.."
    plot_time_series(save_path, brain_data_wh, page, sampling_rate_hilbert, 
                     subject_data, keys=['mind', 'plan', 'anx'], name='ch_average_series')

    cca_amount = min(len(subject_data[0]['bg_header']), brain_data_wh.shape[0])

    # zscore bg data
    bg_data_sc = []
    for row_idx, row in enumerate(bg_data):
        bg_data_sc.append((row - np.mean(row)) / np.std(row))
        header = subject_data[0]['bg_header']
        fig, ax = plt.subplots()
        fig.suptitle(str(header[row_idx]))
        ax.hist(row)
        if save_path:
            norm_dir = os.path.join(save_path, "normality")
            if not os.path.exists(norm_dir):
                os.makedirs(norm_dir)
            fig.savefig(os.path.join(norm_dir, str(header[row_idx]) + '.png'), dpi=310)
    bg_data_sc = np.array(bg_data_sc)

    def do_cca(subject_idxs, cca_impl='statsmodels', squeeze=True):
        # task_idxs = []
        task_brain_data = []
        task_bg_data = []

        for subject_idx in subject_idxs:
            subject = subject_data[subject_idx]
            intervals = subject['intervals']
            start = subject['start']
            name = subject['name']
            # pick only task-data
            for ival_idx, (key, ivals) in enumerate(intervals.items()):
                for ival in ivals:
                    # print str(key), str(ival), str(len(bg_data[0]))
                    x1 = int((start + ival[0]) * sampling_rate_hilbert)
                    x2 = int((start + ival[1]) * sampling_rate_hilbert)
                    # task_idxs.extend(range(x1, x2))

                    task_brain_data.append(np.mean(brain_data_wh[:, range(x1, x2)], axis=1))
                    task_bg_data.append(np.mean(bg_data_sc[:, range(x1, x2)], axis=1))

        task_brain_data = np.array(task_brain_data).T
        task_bg_data = np.array(task_bg_data).T

        # task_brain_data = brain_data_wh[:, task_idxs]
        # task_bg_data = bg_data_sc[:, task_idxs]

        if cca_impl == 'statsmodels': 
            print "Using CCA from statsmodels."
            cancorr = CanCorr(task_brain_data.T, task_bg_data.T)

            brain_weights = cancorr.y_cancoef.T
            bg_weights = cancorr.x_cancoef.T
            brain_scores = np.dot(brain_weights, task_brain_data)
            bg_scores = np.dot(bg_weights, task_bg_data)

            print cancorr.corr_test().summary()
        elif cca_impl == 'sklearn':
            print "Using CCA from sklearn."
            cca = CCA(n_components=cca_amount)
            cca.fit(task_brain_data.T, task_bg_data.T)

            brain_weights = cca.x_weights_.T
            bg_weights = cca.y_weights_.T
            brain_scores = cca.x_scores_.T
            bg_scores = cca.y_scores_.T
        else:
            raise Exception('unknown cca_impl')

        return (brain_weights,
                brain_scores,
                bg_weights,
                bg_scores)

    sm_brain_weights, sm_brain_scores, sm_bg_weights, sm_bg_scores = do_cca(
        range(len(subject_data)))

    sl_brain_weights, sl_brain_scores, sl_bg_weights, sl_bg_scores = do_cca(
        range(len(subject_data)),
        cca_impl='sklearn')

    print "Correlation coefficients for sm: "
    for cca_idx in range(cca_amount):
        print str(cca_idx+1)
        print str(np.corrcoef(sm_brain_scores[cca_idx], sm_bg_scores[cca_idx])[0,1])

    print "Correlation coefficients for sl: "
    for cca_idx in range(cca_amount):
        print str(cca_idx+1)
        print str(np.corrcoef(sl_brain_scores[cca_idx], sl_bg_scores[cca_idx])[0,1])

    print "Weights for sm: "
    for cca_idx in range(cca_amount):
        print str(cca_idx+1)
        print "Brain: " 
        print str(sm_brain_weights[cca_idx])
        print "Bg: " 
        print str(sm_bg_weights[cca_idx])

    print "Weights for sl: "
    for cca_idx in range(cca_amount):
        print str(cca_idx+1)
        print "Brain: " 
        print str(sl_brain_weights[cca_idx])
        print "Bg: " 
        print str(sl_bg_weights[cca_idx])

    # brain_weights_bs = []
    # brain_scores_bs = []
    # bg_weights_bs = []
    # bg_scores_bs = []
    # print "Bootstrapping"
    # for bootstrap_idx in range(bootstrap_iterations):
    #     if bootstrap_idx % 10 == 0:
    #         print str(bootstrap_idx+1) + 'th bootstrap.'
    #     subject_count = len(subject_data)
    #     bs_idxs = np.random.choice(range(subject_count), 
    #                                size=subject_count,
    #                                replace=True)
    # 
    #     results = do_cca(bs_idxs)
    # 
    #     # for weights we will do a (perhaps little biased) trick to get over sign ambiguity
    #     brain_weights_adjusted = []
    #     for cca_idx in range(cca_amount):
    #         if np.corrcoef(brain_weights[cca_idx], results[0][cca_idx])[0,1] > np.corrcoef(brain_weights[cca_idx], -results[0][cca_idx])[0,1]:
    #             brain_weights_adjusted.append(results[0][cca_idx])
    #         else:
    #             brain_weights_adjusted.append(-results[0][cca_idx])
    #     brain_weights_bs.append(np.array(brain_weights_adjusted))
    #
    #     bg_weights_adjusted = []
    #     for cca_idx in range(cca_amount):
    #         if np.corrcoef(bg_weights[cca_idx], results[2][cca_idx])[0,1] > np.corrcoef(bg_weights[cca_idx], -results[2][cca_idx])[0,1]:
    #             bg_weights_adjusted.append(results[2][cca_idx])
    #         else:
    #             bg_weights_adjusted.append(-results[2][cca_idx])
    #     bg_weights_bs.append(np.array(bg_weights_adjusted))
    # 
    #     brain_scores_bs.append(results[1])
    #     bg_scores_bs.append(results[3])
    #
    # brain_weights_upper = np.percentile(brain_weights_bs, 97.5, axis=0)
    # brain_weights_lower = np.percentile(brain_weights_bs, 2.5, axis=0)
    # bg_weights_upper = np.percentile(bg_weights_bs, 97.5, axis=0)
    # bg_weights_lower = np.percentile(bg_weights_bs, 2.5, axis=0)
    # 
    # print "CCA statistics, correlations:"
    # for cca_idx in range(cca_amount):
    #     correlations = [np.corrcoef(brain_scores_bs[bs_idx][cca_idx], bg_scores_bs[bs_idx][cca_idx])[0, 1]
    #                     for bs_idx in range(bootstrap_iterations)]
    #     print str(np.percentile(correlations, 2.5, axis=0)), str(np.mean(correlations, axis=0)), str(np.percentile(correlations, 97.5, axis=0))
    #
    # print "CCA statistics, brain weights:"
    # for iv in range(brain_weights.shape[0]):
    #     print str(iv+1) + '. pair:'
    #     for jv in range(brain_weights.shape[1]):
    #         print str(jv+1), str(brain_weights_lower[iv, jv]), 
    #         print str(brain_weights[iv, jv]), str(brain_weights_upper[iv, jv])

    # print "CCA statistics, bg weights:"
    # for iv in range(bg_weights.shape[0]):
    #     print str(iv+1) + '. pair:'
    #     for jv in range(bg_weights.shape[1]):
    #         print str(jv+1), str(bg_weights_lower[iv, jv]), 
    #         print str(bg_weights[iv, jv]), str(bg_weights_upper[iv, jv])

    def ica_cca():
        # bg_pca = sklearn.decomposition.PCA(whiten=True)
        # bg_pca.fit(bg_data.T)
        # bg_pca_unmixing = bg_pca.components_
        # bg_pca_mixing = np.linalg.pinv(bg_pca_unmixing)
        # bg_data_wh = np.dot(bg_pca_unmixing, bg_data)

        stacked_data_wh = np.vstack([brain_data_wh, bg_data_wh])

        print "Shape before ICA: " + str(stacked_data_wh.shape)

        print "Transforming with ICA.."

        print "Fitting ica to concatenated whitened data."
        ica_params = {
            'n_components': 16,
            'algorithm': 'parallel',
            'whiten': True,
            'max_iter': 10000,
            'tol': 0.000000001
        }
        ica = FastICA(**ica_params)
        ica.fit(stacked_data_wh.T)
        unmixing = ica.components_
        data = np.dot(unmixing, stacked_data_wh)
        mixing = np.linalg.pinv(unmixing)

        brain_weights = []
        bg_weights = []

        for column in np.rollaxis(mixing, axis=1):
            brain_weights.append(
                column[0:brain_ica_mixing.shape[1]])
            bg_weights.append(np.dot(bg_pca_mixing,
                column[brain_ica_mixing.shape[1]:]))

        brain_weights = np.array(brain_weights)
        bg_weights = np.array(bg_weights)

        # sort according to explained variance
        idxs = np.argsort(-np.array(
            [sum([elem**2 for elem in column]) for column
             in np.rollaxis(mixing, axis=1)]))

        brain_weights = brain_weights[idxs][:cca_amount]
        bg_weights = bg_weights[idxs][:cca_amount]

    print "Plotting barchart"
    plot_barcharts(save_path, sl_bg_weights, 'sl_bg_weights',
                   subject_data[0]['bg_header'])
    plot_barcharts(save_path, sl_brain_weights, 'sl_brain_weights', 
                   ['C' + str(elem+1).zfill(2) for elem in range(sl_brain_weights.shape[1])])
    plot_barcharts(save_path, sm_bg_weights, 'sm_bg_weights',
                   subject_data[0]['bg_header'])
    plot_barcharts(save_path, sm_brain_weights, 'sm_brain_weights', 
                   ['C' + str(elem+1).zfill(2) for elem in range(sm_brain_weights.shape[1])])

    for cca_idx in range(sl_bg_scores.shape[0]):
        fig, ax = plt.subplots()
        ax.scatter(sl_bg_scores[cca_idx], sl_brain_scores[cca_idx])
        if save_path:
            scatter_dir = os.path.join(save_path, "scatter")
            if not os.path.exists(scatter_dir):
                os.makedirs(scatter_dir)
            fig.savefig(os.path.join(scatter_dir, 'cca_' + str(cca_idx+1) + '.png'), dpi=620)

    import pdb; pdb.set_trace()
    raise Exception('Kissa')

    ## Classification

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

    # plot_and_regress(keys=['mind', 'anx'])
    # plot_and_regress(keys=['anx', 'plan'])
    # plot_and_regress(keys=['mind', 'plan'])

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
