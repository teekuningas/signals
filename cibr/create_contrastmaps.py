PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('figure', max_open_warning=0)
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

from signals.cibr.common import create_vol_stc
from signals.cibr.common import plot_vol_stc_brainmap


def preprocess(raw, band, min_duration=2, verbose=False):
    if verbose:
        print("Preprocessing.")

    events = mne.find_events(raw, shortest_event=1, min_duration=min_duration/raw.info['sfreq'], uint_cast=True, verbose='warning')
    picks = mne.pick_types(raw.info, meg='grad')
    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names']) 
                       if idx not in picks])

    raw.filter(l_freq=band[0], h_freq=band[1])

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


def extract_intervals_meditation(events, sfreq, first_samp, tasks):
    """ meditation intervals """
    intervals = OrderedDict()

    trigger_info = [
        ('mind', 10),
        ('rest', 11),
        ('plan', 12),
        ('anx', 13),
    ]

    for idx, event in enumerate(events):
        for name, event_id in trigger_info:
            if name not in tasks:
                continue
            if name not in intervals:
                intervals[name] = []
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


def plot_topomap_difference(data_1, data_2, info, factor=1.0):
    data_1 = data_1.copy()
    data_2 = data_2.copy()

    from mne.channels.layout import (_merge_grad_data, find_layout,
                                     _pair_grad_sensors)
    picks, pos = _pair_grad_sensors(info, find_layout(info))
    data_1 = _merge_grad_data(data_1[picks], method='rms').reshape(-1)
    data_2 = _merge_grad_data(data_2[picks], method='rms').reshape(-1)
    
    data = data_2 - data_1

    vmax = np.max(np.abs(data)) / factor
    vmin = -vmax

    fig, ax = plt.subplots()
    mne.viz.topomap.plot_topomap(data, pos, axes=ax, vmin=vmin, vmax=vmax)

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw')
    parser.add_argument('--empty', nargs='+')
    parser.add_argument('--save_path')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    src_method = 'vol'
    mne_method, mne_depth = 'dSPM', None

    sampling_rate_raw = 100.0

    # # FDMSA:
    # tasks = ['heart', 'tone']

    # Meditaatio:
    tasks = ['plan', 'anx']

    preproc_filter = (1, 40)

    # computation_method = 'hilbert'
    computation_method = 'psd'

    band = ('alpha', (7, 14))
    # band = ('beta', (17, 25))

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
        raw, _ = preprocess(raw, band=preproc_filter)
        empty_raws.append(raw)
    empty_raw = mne.concatenate_raws(empty_raws)
    empty_raws = []

    print("Creating sensor covariance matrix..")
    noise_cov = mne.compute_raw_covariance(
        empty_raw, 
        method='empirical')

    path = cli_args.raw

    folder = os.path.dirname(path)
    fname = os.path.basename(path)

    print("Handling " + path)

    raw = mne.io.Raw(path, preload=True, verbose='error')
    raw.resample(sampling_rate_raw)
    raw, events = preprocess(raw, band=preproc_filter, min_duration=1)

    # Meditaatio:
    intervals = extract_intervals_meditation(
        events, 
        raw.info['sfreq'], 
        raw.first_samp,
        tasks)
    subject_name = '_'.join(fname.split('_')[:2])

    # # FDMSA:
    # intervals = extract_intervals_fdmsa_ic(events, raw.info['sfreq'],
    #                                        raw.first_samp)
    # code = fname.split('_tsss')[0].split('IC')[-1][2:]
    # subject_name = 'FDMSA_' + code

    print("Using MRI subject: " + subject_name)

    trans = os.path.join(folder, subject_name + '-trans.fif')

    stc = create_vol_stc(
        raw=raw, 
        trans=trans, 
        subject=subject_name, 
        noise_cov=noise_cov, 
        spacing=vol_spacing,
        mne_method=mne_method,
        mne_depth=mne_depth,
        subjects_dir=subjects_dir) 

    vertices = stc.vertices

    task_stc = {}
    task_sensor = {}

    if computation_method == 'psd':
        print("Prepare using PSD..")
        for key, ivals in intervals.items():
            stc_blocks = []
            sensor_blocks = []
            for ival in ivals:
                start = int(ival[0]*raw.info['sfreq'])
                end = int(ival[1]*raw.info['sfreq'])
                sensor_blocks.append(raw._data[:, start:end])
                stc_blocks.append(stc.data[:, start:end])

            stc_data = np.concatenate(stc_blocks, axis=1)
            sensor_data = np.concatenate(sensor_blocks, axis=1)

            stc_psd, freqs = mne.time_frequency.psd_array_welch(
                stc_data, 
                sfreq=raw.info['sfreq'],
                n_fft=int(raw.info['sfreq']*2))
            sensor_psd, freqs = mne.time_frequency.psd_array_welch(
                sensor_data, 
                sfreq=raw.info['sfreq'], 
                n_fft=int(raw.info['sfreq']*2))

            freqs_idxs = np.where((freqs >= band[1][0]) & (freqs <= band[1][1]))[0]

            task_stc[key] = np.mean(stc_psd[:, freqs_idxs], axis=1)
            task_sensor[key] = np.mean(sensor_psd[:, freqs_idxs], axis=1)
    elif computation_method == 'hilbert':
        print("Prepare using hilbert")
        stc_filtered_data = mne.filter.filter_data(
            stc.data, raw.info['sfreq'], l_freq=band[1][0], h_freq=band[1][1])
        stc_data_hilbert = prepare_hilbert(
            stc_filtered_data, sampling_rate_raw, sampling_rate_hilbert)
        sensor_filtered_data = mne.filter.filter_data(
            raw._data, raw.info['sfreq'], l_freq=band[1][0], h_freq=band[1][1])
        sensor_data_hilbert = prepare_hilbert(
            sensor_filtered_data, sampling_rate_raw, sampling_rate_hilbert)

        for key, ivals in intervals.items():

            stc_blocks = []
            sensor_blocks = []

            length = 2
            for ival in ivals:

                subivals = [(istart*sampling_rate_hilbert, (istart + length)*sampling_rate_hilbert) 
                            for istart in range(int(ival[0]), int(ival[1]), length)]
                for subival in subivals:
                    start = int(subival[0])
                    end = int(subival[1])

                    if end >= stc_data_hilbert.shape[-1]:
                        continue

                    stc_blocks.append(np.mean(stc_data_hilbert[:, start:end], axis=1))
                    sensor_blocks.append(np.mean(sensor_data_hilbert[:, start:end], axis=1))

            task_stc[key] = np.mean(stc_blocks, axis=0)
            task_sensor[key] = np.mean(sensor_blocks, axis=0)
    else:
        raise Exception('Not implemented')

    if save_path:
        activation_topomaps_path = os.path.join(save_path, 'activation_topomaps')
        contrast_topomaps_path = os.path.join(save_path, 'contrast_topomaps')
        activation_data_path = os.path.join(save_path, 'activation_data')
        contrast_data_path = os.path.join(save_path, 'contrast_data')

        try:
            os.makedirs(activation_topomaps_path)
            os.makedirs(contrast_topomaps_path)
            os.makedirs(activation_data_path)
            os.makedirs(contrast_data_path)
        except FileExistsError:
            pass

    # every sensor state separately
    for key, data in task_sensor.items():
        name = subject_name + '_' + key + '_sensor'
        fig = plot_topomap_difference(np.zeros(data.shape), data, raw.info)
        if save_path:
            fig.savefig(os.path.join(activation_topomaps_path, name + '.png'))

    # sensor contrasts
    done = []
    for key_1, data_1 in task_sensor.items():
        for key_2, data_2 in task_sensor.items():
            if (key_1, key_2) in done or (key_2, key_1) in done or key_1 == key_2:
                continue
            done.append((key_1, key_2))

            name = subject_name + '_' + key_1 + '_' + key_2 + '_sensor'
            fig = plot_topomap_difference(data_1, data_2, raw.info)
            if save_path:
                fig.savefig(os.path.join(contrast_topomaps_path, name + '.png'))

    # every stc state separately
    for key, data in task_stc.items():
        name = subject_name + '_' + key + '_stc'
        plot_vol_stc_brainmap(save_path, name, 
            data, vertices, vol_spacing, subjects_dir,
            folder_name="activation_brainmaps") 

    # stc contrasts
    done = []
    for key_1, data_1 in task_stc.items():
        for key_2, data_2 in task_stc.items():
            if (key_1, key_2) in done or (key_2, key_1) in done or key_1 == key_2:
                continue
            done.append((key_1, key_2))
            mean = data_2 - data_1

            name = subject_name + '_' + key_1 + '_' + key_2 + '_stc'
            plot_vol_stc_brainmap(save_path, name, 
                mean, vertices, vol_spacing, subjects_dir, 
                folder_name="contrast_brainmaps")

    # save stc means to file
    if save_path:
        for key, data in task_stc.items():
            name = subject_name + '_' + key + '_stc'
            with open(os.path.join(activation_data_path, name + '.csv'), 'w') as f:
                f.write(', '.join([str(elem) for elem in vertices.tolist()]) + '\n')
                f.write(', '.join([str(elem) for elem in data.tolist()]))

    # save stc contrasts to file

    if save_path:
        done = []
        for key_1, data_1 in task_stc.items():
            for key_2, data_2 in task_stc.items():
                if (key_1, key_2) in done or (key_2, key_1) in done or key_1 == key_2:
                    continue
                done.append((key_1, key_2))
                mean = data_2 - data_1

                name = subject_name + '_' + key_1 + '_' + key_2 + '_stc'
                with open(os.path.join(contrast_data_path, name + '.csv'), 'w') as f:
                    f.write(', '.join([str(elem) for elem in vertices.tolist()]) + '\n')
                    f.write(', '.join([str(elem) for elem in mean.tolist()]))

    print("Hooray!")

