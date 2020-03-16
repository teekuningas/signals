PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('figure', max_open_warning=0)
    matplotlib.use('Agg')
    matplotlib.rc('font', size=3)

import pyface.qt

import sys
import csv
import argparse
import os

from collections import OrderedDict
from itertools import groupby

import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM

from scipy.signal import hilbert
from scipy.signal import decimate
import scipy.fftpack as fftpack

from signals.cibr.lib.stc import create_vol_stc
from signals.cibr.lib.stc import plot_vol_stc_brainmap

from signals.cibr.lib.hmm import interval_length
from signals.cibr.lib.hmm import fractional_occupancy
from signals.cibr.lib.hmm import dwell_time
from signals.cibr.lib.hmm import plot_state_series
from signals.cibr.lib.hmm import plot_task_comparison


def preprocess(raw, band, include_mag=False, verbose=False):
    if verbose:
        print("Preprocessing.")

    events = mne.find_events(raw, shortest_event=1, min_duration=1/raw.info['sfreq'], uint_cast=True, verbose='warning')

    if include_mag:
        picks = mne.pick_types(raw.info, meg=True)
    else:
        picks = mne.pick_types(raw.info, meg='grad')

    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names']) 
                       if idx not in picks])

    if band == 'alpha':
        raw.filter(l_freq=8, 
                   h_freq=13,
                   h_trans_bandwidth=1,
                   l_trans_bandwidth=1)
    elif band == 'beta':
        raw.filter(l_freq=18, 
                   h_freq=25,
                   h_trans_bandwidth=1.5,
                   l_trans_bandwidth=1.5)

    return raw, events


def extract_intervals_fdmsa_ic(events, sfreq, first_samp):
    """ meditation intervals """
    intervals = OrderedDict()

    trigger_info = [
        ('heart', 15),
        ('note', 16),
    ]

    # add 16s

    for name, event_id in trigger_info:
        intervals[name] = []

    counter = 0
    for idx, event in enumerate(events):
        for name, bit in trigger_info:
            if int(format(event[2], '#020b')[-bit]) == 1:
                print(
                    str(format(event[2], '#020b')) + ', ' +
                    str(bit) + ', ' +
                    str(event))
                ival_start = event[0] + 1*sfreq
                ival_end = ival_start + 15*sfreq

                intervals[name].append((
                    (ival_start - first_samp) / sfreq,
                    (ival_end - first_samp) / sfreq))
                counter += 1
    if counter != 16: 
        print("Warning!!! " + str(counter) + " events found.")

    return intervals


def get_amount_of_components(data, explained_var):

    pca = PCA(whiten=True, copy=True)

    data = pca.fit_transform(data.T)

    n_components = np.sum(pca.explained_variance_ratio_.cumsum() <=
                           explained_var)
    return n_components


def fast_hilbert(x):
    return hilbert(x, fftpack.next_fast_len(x.shape[-1]))[..., :x.shape[-1]]


def prepare_hilbert(data, sampling_rate_raw):
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

    return env


def plot_topomap_difference(data_1, data_2, info, ax, factor=1.0):
    data_1 = data_1.copy()
    data_2 = data_2.copy()

    from mne.channels.layout import (_merge_grad_data, find_layout,
                                     _pair_grad_sensors)
    picks, pos = _pair_grad_sensors(info, find_layout(info))
    data_1 = _merge_grad_data(data_1[picks], method='mean').reshape(-1)
    data_2 = _merge_grad_data(data_2[picks], method='mean').reshape(-1)
    
    data = data_2 - data_1

    vmax = np.max(np.abs(data)) / factor
    vmin = -vmax

    mne.viz.topomap.plot_topomap(data, pos, axes=ax, vmin=vmin, vmax=vmax)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', nargs='+')
    parser.add_argument('--empty', nargs='+')
    parser.add_argument('--save-path')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    src_method = 'vol'
    mne_method, mne_depth = 'dSPM', None
    band = 'alpha'
    sampling_rate_raw = 100.0

    include_mag = True

    tasks = ['heart', 'tone']

    surf_spacing = 'ico3'
    vol_spacing = '10'

    # process similarly to input data 
    empty_paths = cli_args.empty
    empty_raws = []
    for fname in empty_paths:
        raw = mne.io.Raw(fname, preload=True, verbose='error')
        raw.resample(sampling_rate_raw)
        raw, _ = preprocess(raw, band=band, include_mag=include_mag)
        empty_raws.append(raw)
    empty_raw = mne.concatenate_raws(empty_raws)
    empty_raws = []

    print("Creating sensor covariance matrix..")
    noise_cov = mne.compute_raw_covariance(
        empty_raw, 
        method='empirical')

    subjects = []
    
    for raw_fname in cli_args.raw:

        path = raw_fname

        folder = os.path.dirname(path)
        fname = os.path.basename(path)

        # subject = '_'.join(fname.split('_')[:2])
        code = fname.split('_tsss')[0].split('IC')[-1][2:]
        subject = 'FDMSA_' + code

        identifier = fname.split('_tsss')[0] 

        print("Using MRI subject: " + subject)

        trans = os.path.join(folder, subject + '-trans.fif')

        print("Handling " + path)

        raw = mne.io.Raw(path, preload=True, verbose='error')
        raw.resample(sampling_rate_raw)
        raw, events = preprocess(raw, band=band, include_mag=include_mag, min_duration=1)

        intervals = extract_intervals_fdmsa_ic(events, raw.info['sfreq'],
                                               raw.first_samp)

        stc = create_vol_stc(
            raw=raw, 
            trans=trans, 
            subject=subject, 
            noise_cov=noise_cov, 
            spacing=vol_spacing,
            mne_method=mne_method,
            mne_depth=mne_depth,
            subjects_dir=subjects_dir) 

        subject_data = {}

        subject_data['name'] = identifier
        subject_data['intervals'] = intervals

        print("Prepare using hilbert")
        stc_data = prepare_hilbert(stc.data, sampling_rate_raw)
        subject_data['stc_data'] = stc_data

        sensor_data = prepare_hilbert(raw._data, sampling_rate_raw)
        subject_data['sensor_data'] = sensor_data

        subjects.append(subject_data)

        vertices = stc.vertices

        sfreq = raw.info['sfreq']

    concatenated_stc = []
    concatenated_sensor = []

    factor = sfreq / 10
    sfreq = 10

    normalization = 'demean_division'
    data_type = 'stc'

    current_time = 0
    for subject in subjects:
        sensor_data = np.array([decimate(row, int(factor)) for row in subject['sensor_data']])
        stc_data = np.array([decimate(row, int(factor)) for row in subject['stc_data']])

        # sensor_data = subject['sensor_data']
        # stc_data = subject['stc_data']

        if normalization == 'note':
            sensor_baseline = []
            stc_baseline = []
            for key, ivals in subject['intervals'].items():
                if key != "note":
                    continue
                for ival in ivals:
                    start_idx = int(ival[0] * sfreq)
                    end_idx = int(ival[1] * sfreq)
                    sensor_baseline.append(sensor_data[:, start_idx:end_idx])
                    stc_baseline.append(stc_data[:, start_idx:end_idx])
            sensor_baseline = np.mean(np.mean(sensor_baseline, axis=0), axis=1)
            stc_baseline = np.mean(np.mean(stc_baseline, axis=0), axis=1)
 
            sensor_data -= sensor_baseline[:, np.newaxis]
            stc_data -= stc_baseline[:, np.newaxis]
 
        elif normalization == 'demean':
            sensor_data -= np.mean(sensor_data, axis=1)[:, np.newaxis]
            stc_data -= np.mean(stc_data, axis=1)[:, np.newaxis]
        elif normalization == 'demean_division':
            sensor_data /= np.mean(sensor_data, axis=1)[:, np.newaxis]
            stc_data /= np.mean(stc_data, axis=1)[:, np.newaxis]

        elif normalization == 'zscore_sep':
            sensor_data = (sensor_data - np.mean(sensor_data, axis=1)[:, np.newaxis]) / np.std(sensor_data, axis=1)[:, np.newaxis]
            stc_data = (stc_data - np.mean(stc_data, axis=1)[:, np.newaxis]) / np.std(stc_data, axis=1)[:, np.newaxis]
        elif normalization == 'zscore':
            sensor_data = (sensor_data - np.mean(sensor_data)) / np.std(sensor_data)
            stc_data = (stc_data - np.mean(stc_data)) / np.std(stc_data)

        concatenated_sensor.append(sensor_data)
        concatenated_stc.append(stc_data)

        subject['start_time'] = current_time
        current_time += len(sensor_data[1]) / sfreq

    concatenated_sensor = np.concatenate(concatenated_sensor, axis=1)
    concatenated_stc = np.concatenate(concatenated_stc, axis=1)

    if data_type == 'stc':
        data = concatenated_stc
    else:
        data = concatenated_sensor

    n_components = 12
    n_states = 4

    pca = PCA(n_components=n_components, whiten=True)
    comps = pca.fit_transform(np.array(data.T)).T

    markov = GaussianHMM(
        n_components=n_states,
        covariance_type='full',
        verbose=True,
        n_iter=10000)

    print("Fitting HMM")
    markov.fit(comps.T)

    for state_idx in range(n_states):
        if not save_path:
            continue
        weights = markov.means_[state_idx]
        statemap = np.dot(weights, pca.components_) + pca.mean_
        cov = markov.covars_[state_idx]
        fig, axes = plt.subplots(ncols=2, nrows=1, gridspec_kw={'width_ratios': [6, 2]})

        if data_type == 'stc':
            plot_vol_stc_brainmap(None, 'state_' + str(state_idx+1).zfill(2), statemap,
                                  vertices, vol_spacing, subjects_dir, axes=axes[0])
        else:
            plot_topomap_difference(np.zeros(statemap.shape), statemap, raw.info, axes[0])

        labels = ['Component ' + str(lidx).zfill(3) for lidx in range(n_components)]
        sns.heatmap(cov, ax=axes[1], xticklabels=labels, yticklabels=labels, square=True,
                cmap='RdBu_r', center=0.0, cbar_kws={'shrink': .50,
                                                     'aspect': 10})

        state_path = os.path.join(save_path, 'states')
        if not os.path.exists(state_path):
            os.makedirs(state_path)

        name = 'state_' + str(state_idx+1).zfill(2)
        path = os.path.join(state_path, name + '.png')
        fig.savefig(path, dpi=620)


    print("Inferring hidden states")
    state_chain = markov.predict_proba(comps.T)

    state_chain_by_subject = []
    for subject_idx, subject in enumerate(subjects):
        subject_start = subject['start_time']

        # find out first sample of the current subject in concatenated data
        start = int(subject_start * sfreq)

        # find out last sample of the current subject in concatenated data
        if subject_idx != len(subjects) - 1:
            end = int(subjects[subject_idx + 1]['start_time'] * sfreq)
        else:
            end = data.shape[1]

        state_chain_by_subject.append(state_chain[start:end, :])

    task_annotations = []
    for subject in subjects:
        subject_annot = []
        for key, ivals in subject['intervals'].items():
            if key == 'heart':
                subject_annot.append((key, 'red', ivals))
            else:
                subject_annot.append((key, 'blue', ivals))
        task_annotations.append(subject_annot)

    timeblocks = [
        ('full', (0, 600)), 
        ('part01', (0, 50)), 
        ('part02', (50, 100)), 
        ('part03', (100, 150)), 
        ('part04', (150, 200)), 
        ('part05', (200, 250)), 
        ('part06', (250, 300)), 
        ('part07', (300, 350)),
        ('part08', (350, 400)), 
        ('part09', (400, 450)), 
        ('part10', (450, 500)),
        ('part11', (500, 550)),
        ('part12', (550, 600))
    ]

    timeblocks = [('full', (0, 600))]

    # make subjectblocks of 8 subjects
    sb_size = 8
    subjectblocks = []
    for idx in range(len(subjects) // sb_size):
        subjectblocks.append([ii for ii in range(sb_size*idx, sb_size*(idx+1))])
    if len(subjects) % sb_size != 0:
        subjectblocks.append([ii for ii in range(sb_size*(len(subjects) // sb_size), len(subjects))])

    for block_idx, subjectblock in enumerate(subjectblocks):
        print("Computing " + str(block_idx+1) + " subjectblock")
        state_chains = [chain for idx, chain in enumerate(state_chain_by_subject)
                        if idx in subjectblock]
        names = [subject['name'] for idx, subject in enumerate(subjects)
                 if idx in subjectblock]
        annotations = [annot for idx, annot in enumerate(task_annotations)
                       if idx in subjectblock]

        for timeblock_name, (tmin, tmax) in timeblocks:
            if not save_path:
                continue

            print("Plotting timeblock from " + str(tmin) + " to " + str(tmax))
            fig = plot_state_series(state_chains, 
                                    tmin, tmax, sfreq,
                                    names, 
                                    task_annotations=annotations,
                                    probabilistic=False)

            series_path = os.path.join(save_path, 'series')
            if not os.path.exists(series_path):
                os.makedirs(series_path)

            name = timeblock_name + '_subs' + str(block_idx+1).zfill(2)
            path = os.path.join(series_path, name + '.png')
            fig.savefig(path, dpi=620)

    columns = ['subject', 'occ_freq_heart', 'dwell_heart', 'interval_heart',
               'occ_freq_note', 'dwell_note', 'interval_note']
    columns = ['subject']
    for state_idx in range(n_states):
        columns.append('state_' + str(state_idx+1).zfill(2) + '_occ_freq_heart')
        columns.append('state_' + str(state_idx+1).zfill(2) + '_dwell_heart')
        columns.append('state_' + str(state_idx+1).zfill(2) + '_interval_heart')
        columns.append('state_' + str(state_idx+1).zfill(2) + '_occ_freq_note')
        columns.append('state_' + str(state_idx+1).zfill(2) + '_dwell_note')
        columns.append('state_' + str(state_idx+1).zfill(2) + '_interval_note')
    data = []
    for subject_idx, subject in enumerate(subjects):
        subject_vals = [subject['name']]
        for state_idx in range(n_states):
            state_chain = state_chain_by_subject[subject_idx]

            ival_chain = []
            for ival in subject['intervals']['heart']:
                start_idx = int(ival[0] * sfreq)
                end_idx = int(ival[1] * sfreq)
                ival_chain.append(state_chain[start_idx:end_idx])
            ival_chain = np.concatenate(ival_chain, axis=0)

            subject_vals.append(fractional_occupancy(ival_chain, sfreq)[state_idx])
            subject_vals.append(dwell_time(ival_chain, sfreq)[state_idx])
            subject_vals.append(interval_length(ival_chain, sfreq)[state_idx])

            ival_chain = []
            for ival in subject['intervals']['note']:
                start_idx = int(ival[0] * sfreq)
                end_idx = int(ival[1] * sfreq)
                ival_chain.append(state_chain[start_idx:end_idx])
            ival_chain = np.concatenate(ival_chain, axis=0)

            subject_vals.append(fractional_occupancy(ival_chain, sfreq)[state_idx])
            subject_vals.append(dwell_time(ival_chain, sfreq)[state_idx])
            subject_vals.append(interval_length(ival_chain, sfreq)[state_idx])

        data.append(subject_vals)

    df = pd.DataFrame(data, columns=columns)
    for stat_name in ['occ_freq', 'dwell', 'interval']:
        for state_idx in range(n_states):
            heart_key = 'state_' + str(state_idx+1).zfill(2) + '_' + stat_name + '_heart'
            note_key = 'state_' + str(state_idx+1).zfill(2) + '_' + stat_name + '_note'
            from scipy.stats import ttest_1samp
            result = ttest_1samp(df[heart_key] - df[note_key], 0)
            print("T-test result for " + stat_name + " (state " + 
                  str(state_idx+1).zfill(2) + "): " + str(result))

    # do group comparisons later..

    # plot chain stats
    for subject_idx, subject in enumerate(subjects):

        state_chain = state_chain_by_subject[subject_idx]

        # plot fractional occupancies
        title = "Frac occ of " + subject['name']
        ylabel = 'Fractional occupancy'
        occ_fig = plot_task_comparison(state_chain, sfreq, subject['intervals'].items(),
                                       fractional_occupancy, ylabel=ylabel, title=title)

        # plot dwell times
        title = "Dwell times of " + subject['name']
        ylabel = 'Average dwell time'
        dwell_fig = plot_task_comparison(state_chain, sfreq, subject['intervals'].items(),
                                         dwell_time, ylabel=ylabel, title=title)

        # plot ival times
        title = "Interval lengths of " + subject['name']
        ylabel = 'Average interval'
        ival_fig = plot_task_comparison(state_chain, sfreq, subject['intervals'].items(),
                                        interval_length, ylabel=ylabel, title=title)
 
    
        if save_path:
            chain_stats_path = os.path.join(save_path, 'chain_stats')
            if not os.path.exists(chain_stats_path):
                os.makedirs(chain_stats_path)

            fname = 'frac_occ_' + subject['name']
            path = os.path.join(chain_stats_path, fname + '.png')
            occ_fig.savefig(path, dpi=155)

            fname = 'dwell_time_' + subject['name']
            path = os.path.join(chain_stats_path, fname + '.png')
            dwell_fig.savefig(path, dpi=155)

            fname = 'interval_length_' + subject['name']
            path = os.path.join(chain_stats_path, fname + '.png')
            ival_fig.savefig(path, dpi=155)
 
