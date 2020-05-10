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

import nibabel as nib
import mne
import numpy as np
import matplotlib.pyplot as plt 
import sklearn

from scipy.signal import hilbert
from scipy.signal import decimate
import scipy.fftpack as fftpack

from signals.cibr.lib.stc import create_vol_stc
from signals.cibr.lib.stc import plot_vol_stc_brainmap
from signals.cibr.lib.sensor import plot_sensor_topomap

from signals.cibr.lib.triggers import extract_intervals_meditaatio

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
    parser.add_argument('--raw')
    parser.add_argument('--empty', nargs='+')
    parser.add_argument('--save_path')
    parser.add_argument('--tasks')
    parser.add_argument('--method')
    parser.add_argument('--band')
    parser.add_argument('--compute_stc')
    parser.add_argument('--normalize')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    vol_spacing = '10'
    mne_method, mne_depth = 'dSPM', None

    sampling_rate_raw = 100.0
    prefilter_band = (1, 40)

    compute_stc = True
    if cli_args.compute_stc:
        compute_stc = True if cli_args.compute_stc == 'true' else False

    pointwise_normalization = True
    if cli_args.normalize:
        pointwise_normalization = True if cli_args.normalize == 'true' else False

    # band = (17, 25)
    band = (7, 14)
    if cli_args.band:
        band = (int(cli_args.band.split()[0]), int(cli_args.band.split()[1]))

    # computation_method = 'hilbert'
    computation_method = 'psd'
    if cli_args.method:
        computation_method = cli_args.method

    tasks = ('mind', 'plan')
    if cli_args.tasks:
        tasks = cli_args.tasks.split()[0], cli_args.tasks.split()[1]

    sampling_rate_hilbert = 1.0

    if compute_stc:
        # process similarly to input data 
        empty_paths = cli_args.empty
        empty_raws = []
        for fname in empty_paths:
            raw = mne.io.Raw(fname, preload=True, verbose='error')
            raw.filter(*prefilter_band)
            raw.crop(tmin=(raw.times[0]+3), tmax=raw.times[-1]-3)
            raw.resample(sampling_rate_raw)
            raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names'])
                               if idx not in mne.pick_types(raw.info, meg=True)])
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

    events = mne.find_events(raw, shortest_event=1, min_duration=1/raw.info['sfreq'],
                             uint_cast=True, verbose='warning')

    intervals = extract_intervals_meditaatio(
        events, 
        raw.info['sfreq'], 
        raw.first_samp,
        tasks)
    subject = fname.split('_block')[0]
    identifier = fname.split('_tsss')[0]
    trans = os.path.join(folder, subject + '-trans.fif')

    raw.filter(*prefilter_band)
    raw.resample(sampling_rate_raw)
    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names'])
                       if idx not in mne.pick_types(raw.info, meg=True)])


    if compute_stc:
        stc = create_vol_stc(
            raw=raw, 
            trans=trans, 
            subject=subject, 
            noise_cov=noise_cov, 
            spacing=vol_spacing,
            mne_method=mne_method,
            mne_depth=mne_depth,
            subjects_dir=subjects_dir) 

        vertices = stc.vertices

    task_data = {}

    if computation_method == 'psd':
        print("Prepare using PSD..")

        if pointwise_normalization:
            all_blocks = []
            for key, ivals in intervals.items():
                for ival in ivals:
                    start = int(ival[0]*raw.info['sfreq'])
                    end = int(ival[1]*raw.info['sfreq'])
                    if compute_stc:
                        all_blocks.append(stc.data[:, start:end])
                    else:
                        all_blocks.append(raw._data[:, start:end])

            concatenated = np.concatenate(all_blocks, axis=1)

            psd, freqs = mne.time_frequency.psd_array_welch(
                concatenated, 
                sfreq=raw.info['sfreq'],
                n_fft=int(raw.info['sfreq']*2))

            freqs_idxs = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]

            normalization_data = np.mean(psd[:, freqs_idxs], axis=1)

        for key, ivals in intervals.items():
            task_blocks = []
            for ival in ivals:
                start = int(ival[0]*raw.info['sfreq'])
                end = int(ival[1]*raw.info['sfreq'])
                if compute_stc:
                    task_blocks.append(stc.data[:, start:end])
                else:
                    task_blocks.append(raw._data[:, start:end])

            concatenated = np.concatenate(task_blocks, axis=1)

            psd, freqs = mne.time_frequency.psd_array_welch(
                concatenated, 
                sfreq=raw.info['sfreq'],
                n_fft=int(raw.info['sfreq']*2))

            freqs_idxs = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]

            task_data[key] = np.mean(psd[:, freqs_idxs], axis=1)

            if pointwise_normalization:
                task_data[key] /= np.abs(normalization_data)

    elif computation_method == 'hilbert':
        print("Prepare using hilbert")
        if compute_stc:
            filtered_data = mne.filter.filter_data(
                stc.data, raw.info['sfreq'], l_freq=band[0], h_freq=band[1])
        else:
            filtered_data = mne.filter.filter_data(
                raw._data, raw.info['sfreq'], l_freq=band[0], h_freq=band[1])

        data_hilbert = prepare_hilbert(
            filtered_data, sampling_rate_raw, sampling_rate_hilbert)

        if pointwise_normalization:
            all_blocks = []
            for key, ivals in intervals.items():
                for ival in ivals:
                    start = int(ival[0] * sampling_rate_hilbert)
                    end = int(ival[1] * sampling_rate_hilbert)
                    all_blocks.append(np.mean(data_hilbert[:, start:end], axis=1))
            normalization_data = np.mean(all_blocks, axis=0)

        for key, ivals in intervals.items():
            task_blocks = []
            for ival in ivals:
                start = int(ival[0] * sampling_rate_hilbert)
                end = int(ival[1] * sampling_rate_hilbert)
                task_blocks.append(np.mean(data_hilbert[:, start:end], axis=1))

            task_data[key] = np.mean(task_blocks, axis=0)
            if pointwise_normalization:
                task_data[key] /= np.abs(normalization_data)

    if save_path:
        activation_maps_path = os.path.join(save_path, 'activation_maps')
        contrast_maps_path = os.path.join(save_path, 'contrast_maps')
        activation_data_path = os.path.join(save_path, 'activation_data')
        contrast_data_path = os.path.join(save_path, 'contrast_data')

        try:
            os.makedirs(activation_maps_path)
            os.makedirs(contrast_maps_path)
            os.makedirs(activation_data_path)
            os.makedirs(contrast_data_path)
        except FileExistsError:
            pass

    # plot every state separately
    for key, data in task_data.items():
        name = identifier + '_' + key
        fig, ax = plt.subplots()
        if compute_stc:
            plot_vol_stc_brainmap(data, vertices, vol_spacing, subjects_dir,
                ax) 
        else:
            plot_sensor_topomap(data, raw.info, ax)

        if save_path:
            fig.savefig(os.path.join(activation_maps_path, name + '.png'))

    # plot contrasts
    done = []
    for key_1, data_1 in task_data.items():
        for key_2, data_2 in task_data.items():
            if (key_1, key_2) in done or (key_2, key_1) in done or key_1 == key_2:
                continue
            done.append((key_1, key_2))

            name = identifier + '_' + key_1 + '_' + key_2

            fig, ax = plt.subplots()

            if compute_stc:
                plot_vol_stc_brainmap(data_2 - data_1, vertices, vol_spacing, subjects_dir,
                    ax) 
            else:
                plot_sensor_topomap(data_2 - data_1, raw.info, ax)

            if save_path:
                fig.savefig(os.path.join(contrast_maps_path, name + '.png'))

    if save_path:
        for key, data in task_data.items():
            name = identifier + '_' + key
            with open(os.path.join(activation_data_path, name + '.csv'), 'w') as f:
                if compute_stc:
                    f.write(', '.join([str(elem) for elem in vertices.tolist()]) + '\n')
                    f.write(', '.join([str(elem) for elem in data.tolist()]))
                else:
                    f.write(', '.join([str(elem) for elem in raw.info['ch_names']]) + '\n')
                    f.write(', '.join([str(elem) for elem in data.tolist()]))

    if save_path:
        done = []
        for key_1, data_1 in task_data.items():
            for key_2, data_2 in task_data.items():
                if (key_1, key_2) in done or (key_2, key_1) in done or key_1 == key_2:
                    continue
                done.append((key_1, key_2))
                mean = data_2 - data_1

                name = identifier + '_' + key_1 + '_' + key_2
                
                with open(os.path.join(contrast_data_path, name + '.csv'), 'w') as f:
                    if compute_stc:
                        f.write(', '.join([str(elem) for elem in vertices.tolist()]) + '\n')
                        f.write(', '.join([str(elem) for elem in mean.tolist()]))
                    else:

                        f.write(', '.join([str(elem) for elem in raw.info['ch_names']]) + '\n')
                        f.write(', '.join([str(elem) for elem in mean.tolist()]))

    print("Hooray!")

