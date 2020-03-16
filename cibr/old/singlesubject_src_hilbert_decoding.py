PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('font', size=3)
    matplotlib.use('Agg')

import pyface.qt

import sys
import csv
import argparse
import os

from collections import OrderedDict

import nibabel as nib
import mne
import numpy as np
import matplotlib.pyplot as plt 
import sklearn

from nilearn.plotting import plot_glass_brain

from sklearn.decomposition import FastICA

from scipy.signal import hilbert
from scipy.signal import decimate
import scipy.fftpack as fftpack

# from icasso import Icasso

from signals.cibr.common import preprocess
from signals.cibr.common import create_vol_stc
from signals.cibr.common import plot_vol_stc_brainmap


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


def get_amount_of_components(data, explained_var):

    from sklearn.decomposition import PCA
    pca = PCA(whiten=True, copy=True)

    data = pca.fit_transform(data.T)

    n_components = np.sum(pca.explained_variance_ratio_.cumsum() <=
                           explained_var)
    return n_components


def fast_hilbert(x):
    return hilbert(x, fftpack.next_fast_len(x.shape[-1]))[..., :x.shape[-1]]


def prepare_hilbert(data, sampling_rate_raw,
                    sampling_rate_hilbert):
    # get envelope as abs of analytic signal
    import time
    start = time.time()
    rowsplits = np.array_split(data, 2, axis=0)
    env_rowsplits = []
    for rowsplit in rowsplits:
        blocks = np.array_split(rowsplit, 4, axis=1)
        env_blocks = []
        for block in blocks:
            env_blocks.append(np.abs(fast_hilbert(block)))

        env_rowsplits.append(np.concatenate(env_blocks, axis=1))
    env = np.concatenate(env_rowsplits, axis=0)

    # decimate first with five
    decimated = decimate(env, 5)
    # and then the rest
    factor = int(((sampling_rate_raw / 5.0) / sampling_rate_hilbert))
    decimated = decimate(decimated, factor)
    return decimated


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    parser.add_argument('--contrast')
    parser.add_argument('--raw')
    parser.add_argument('--empty', nargs='+')
    parser.add_argument('--save-path')
    parser.add_argument('--csv-path')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    src_method = 'vol'
    mne_method, mne_depth = 'dSPM', None
    band = (6, 15)
    sampling_rate_raw = 50.0

    if not cli_args.task:
        task = 'mind'
    else:
        task = cli_args.task

    tasks = [task, cli_args.contrast]

    surf_spacing = 'ico3'
    vol_spacing = '10'
    page = 4

    # hilbert ica settings
    sampling_rate_hilbert = 1.0

    # process similarly to input data 
    empty_paths = cli_args.empty
    empty_raws = []
    for fname in empty_paths:
        raw = mne.io.Raw(fname, preload=True, verbose='error')
        raw.resample(sampling_rate_raw)
        raw, _ = preprocess(raw, filter_=band)
        empty_raws.append(raw)
    empty_raw = mne.concatenate_raws(empty_raws)
    empty_raws = []

    print("Creating sensor covariance matrix..")
    noise_cov = mne.compute_raw_covariance(
        empty_raw, 
        method='empirical')

    current_time = 0
    subject_data = {}

    path = cli_args.raw

    folder = os.path.dirname(path)
    fname = os.path.basename(path)

    subject = '_'.join(fname.split('_')[:2])

    print("Using MRI subject: " + subject)

    trans = os.path.join(folder, subject + '-trans.fif')

    print("Handling " + path)

    raw = mne.io.Raw(path, preload=True, verbose='error')
    raw.resample(sampling_rate_raw)
    raw, events = preprocess(raw, filter_=band, min_duration=1)

    intervals = extract_intervals_meditation(
        events, 
        raw.info['sfreq'], 
        raw.first_samp,
        tasks)

    stc = create_vol_stc(
        raw=raw, 
        trans=trans, 
        subject=subject, 
        noise_cov=noise_cov, 
        spacing=vol_spacing,
        mne_method=mne_method,
        mne_depth=mne_depth,
        subjects_dir=subjects_dir) 

    subject_data['name'] = subject
    subject_data['intervals'] = intervals
    subject_data['start'] = current_time

    print("Prepare using hilbert")
    data = prepare_hilbert(stc.data, sampling_rate_raw, sampling_rate_hilbert)
    subject_data['data'] = data
    print("Hilbert done")

    current_time += data.shape[-1] / sampling_rate_hilbert
    vertices = stc.vertices

    explained_var = 0.95
    n_explained_components = get_amount_of_components(subject_data['data'], 
                                                      explained_var)
    n_components = max(min(n_explained_components, 30), 7)

    ica_params = {
        'n_components': n_components,
        'algorithm': 'parallel',
        'whiten': True,
        'max_iter': 100000,
        'tol': 0.0000000001
    }
    ica = FastICA(**ica_params)
    ica.fit(subject_data['data'].T)
    ica_unmixing = ica.components_
    ica_mixing = np.linalg.pinv(ica_unmixing)

    independent_data = np.dot(ica_unmixing, subject_data['data'])

    savename = subject_data['name'] + '_' + tasks[0] + '_' + tasks[1]
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    def plot_and_save(blocktype, interval_dict):
        X = [] 
        y = []
        length = 4
        for key, ivals in interval_dict.items():
            if key not in tasks:
                continue
            for ival in ivals:
                subivals = [(istart, istart + length) for istart in 
                            range(int(ival[0]), int(ival[1]), length)]
                for subival in subivals:
                    start = int(subival[0]*sampling_rate_hilbert)
                    end = int(subival[1]*sampling_rate_hilbert)
                    X.append(np.mean(subject_data['data'][:, start:end], axis=1))
                    if key == tasks[0]:
                        y.append(0)
                    if key == tasks[1]:
                        y.append(1)

        X_ = np.array(X)
        y_ = np.array(y)

        difference_brainmap = np.mean(X_[y_==1], axis=0) - np.mean(X_[y_==0], axis=0)


        plot_vol_stc_brainmap(
            save_path, savename + '_brainmap_' + blocktype, 
            difference_brainmap, vertices, vol_spacing, subjects_dir) 

        if save_path:
            data_path = os.path.join(save_path, 'data')
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            path = os.path.join(data_path, savename + '_' + blocktype + '.csv')

            with open(path, 'w') as f:
                f.write(', '.join([str(elem) for elem in vertices.tolist()]) + '\n')
                f.write(', '.join([str(elem) for elem in difference_brainmap.tolist()]))

    intervals = subject_data['intervals']
    block1_task0 = intervals[tasks[0]][:int(len(intervals[tasks[0]])/2)]
    block1_task1 = intervals[tasks[1]][:int(len(intervals[tasks[1]])/2)]
    block2_task0 = intervals[tasks[0]][int(len(intervals[tasks[0]])/2):]
    block2_task1 = intervals[tasks[1]][int(len(intervals[tasks[1]])/2):]

    block1_intervals = OrderedDict()
    block1_intervals[tasks[0]] = block1_task0
    block1_intervals[tasks[1]] = block1_task1
    block2_intervals = OrderedDict()
    block2_intervals[tasks[0]] = block2_task0
    block2_intervals[tasks[1]] = block2_task1

    # get accuracy by training with session 1 
    # and testing with session 2
    block1_ica_X = [] 
    block1_brain_X = [] 
    block1_y = []
    length = 2
    for key, ivals in block1_intervals.items():
        if key not in tasks:
            continue
        for ival in ivals:
            subivals = [(istart, istart + length) for istart in range(int(ival[0]), int(ival[1]), length)]
            for subival in subivals:
                start = int(subival[0]*sampling_rate_hilbert)
                end = int(subival[1]*sampling_rate_hilbert)
                block1_ica_X.append(np.mean(independent_data[:, start:end], axis=1))
                block1_brain_X.append(np.mean(subject_data['data'][:, start:end], axis=1))
                if key == tasks[0]:
                    block1_y.append(0)
                if key == tasks[1]:
                    block1_y.append(1)
    block2_ica_X = [] 
    block2_brain_X = [] 
    block2_y = []
    length = 2
    for key, ivals in block2_intervals.items():
        if key not in tasks:
            continue
        for ival in ivals:
            subivals = [(istart, istart + length) for istart in range(int(ival[0]), int(ival[1]), length)]
            for subival in subivals:
                start = int(subival[0]*sampling_rate_hilbert)
                end = int(subival[1]*sampling_rate_hilbert)
                block2_ica_X.append(np.mean(independent_data[:, start:end], axis=1))
                block2_brain_X.append(np.mean(subject_data['data'][:, start:end], axis=1))
                if key == tasks[0]:
                    block2_y.append(0)
                if key == tasks[1]:
                    block2_y.append(1)

    block1_ica_X, block1_brain_X, block1_y = sklearn.utils.shuffle(
        np.array(block1_ica_X), np.array(block1_brain_X), np.array(block1_y))
    block2_ica_X, block2_brain_X, block2_y = sklearn.utils.shuffle(
        np.array(block2_ica_X), np.array(block2_brain_X), np.array(block2_y))

    print("Classify first block trained with first block: ")
    clf_block1 = sklearn.linear_model.SGDClassifier(
        loss='log',
        penalty='l2',
        tol=1e-8,
        class_weight='balanced',
    )
    scores_block1 = sklearn.model_selection.cross_val_score(clf_block1, block1_ica_X, block1_y, cv=3)
    train_score_block1 = np.mean(scores_block1)
    print("Results (block1): " + str(scores_block1))
    print("Train CV score (block1): " + str(train_score_block1))

    print("Classify first block trained with first block: ")
    clf_block2 = sklearn.linear_model.SGDClassifier(
        loss='log',
        penalty='l2',
        tol=1e-8,
        class_weight='balanced',
    )
    scores_block2 = sklearn.model_selection.cross_val_score(clf_block2, block2_ica_X, block2_y, cv=3)
    train_score_block2 = np.mean(scores_block2)
    print("Results (block2): " + str(scores_block2))
    print("Train CV score (block2): " + str(train_score_block2))

    clf_block1.fit(block1_ica_X, block1_y)
    val_block1 = clf_block1.score(block2_ica_X, block2_y)

    clf_block2.fit(block2_ica_X, block2_y)
    val_block2 = clf_block2.score(block1_ica_X, block1_y)
    print("Validation scores: " + str(val_block1) + ', ' + str(val_block2))

    validation_score = np.mean([val_block1, val_block2])

    block1_difference_brainmap = (
        np.mean(block1_brain_X[block1_y==1], axis=0) - 
        np.mean(block1_brain_X[block1_y==0], axis=0)
    )

    block2_difference_brainmap = (
        np.mean(block2_brain_X[block2_y==1], axis=0) - 
        np.mean(block2_brain_X[block2_y==0], axis=0)
    )

    import scipy.stats
    pearson = scipy.stats.pearsonr(block1_difference_brainmap, 
                                   block2_difference_brainmap)[0]
    spearman = scipy.stats.spearmanr(block1_difference_brainmap, 
                                     block2_difference_brainmap)[0]

    print("Pearson: " + str(pearson))
    print("Spearman: " + str(spearman))

    similarity_corrcoef = pearson

    print('Similarity coef: ' + str(similarity_corrcoef))

    if save_path:
        info_path = os.path.join(save_path, 'subject_info')
        if not os.path.exists(info_path):
            os.makedirs(info_path)

        path = os.path.join(info_path,
            savename + '.csv')

        with open(path, 'w') as f:
            f.write(', '.join([
                savename,
                str(train_score_block1),
                str(train_score_block2),
                str(validation_score),
                str(similarity_corrcoef)]))

    plot_and_save('both', intervals)
    plot_and_save('block1', block1_intervals)
    plot_and_save('block2', block2_intervals)
    
