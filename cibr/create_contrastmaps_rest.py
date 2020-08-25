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
from signals.cibr.lib.triggers import extract_intervals_meditaatio_rest
from signals.cibr.lib.triggers import extract_intervals_fdmsa_ic

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
    parser.add_argument('--raw_task')
    parser.add_argument('--raw_rest')
    parser.add_argument('--task_cond')
    parser.add_argument('--rest_cond')
    parser.add_argument('--empty', nargs='+')
    parser.add_argument('--save_path')
    parser.add_argument('--band')
    parser.add_argument('--compute_stc')
    parser.add_argument('--depth')
    parser.add_argument('--spacing')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    mne_method = 'dSPM'

    vol_spacing = '10'
    if cli_args.spacing is not None:
        vol_spacing = str(cli_args.spacing)

    mne_depth = None
    if cli_args.depth is not None:
        mne_depth = float(cli_args.depth)

    sampling_rate_raw = 100.0
    prefilter_band = (1, 40)

    compute_stc = True
    if cli_args.compute_stc:
        compute_stc = True if cli_args.compute_stc == 'true' else False

    # band = (17, 25)
    band = (7, 14)
    if cli_args.band:
        band = (int(cli_args.band.split()[0]), int(cli_args.band.split()[1]))

    task_cond = ['mind']
    if cli_args.task_cond:
        task_cond = []
        for cond in cli_args.task_cond.split():
            task_cond.append(cond)

    rest_cond = ['eo']
    if cli_args.rest_cond:
        rest_cond = []
        for cond in cli_args.rest_cond.split():
            rest_cond.append(cond)

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

    task_path = cli_args.raw_task
    rest_path = cli_args.raw_rest

    folder = os.path.dirname(task_path)
    fname = os.path.basename(task_path)

    print("Handling " + task_path)

    raw_task = mne.io.Raw(task_path, preload=True, verbose='error')
    raw_rest = mne.io.Raw(rest_path, preload=True, verbose='error')

    events = mne.find_events(raw_task, shortest_event=1, min_duration=1/raw_task.info['sfreq'],
                             uint_cast=True, verbose='warning')

    task_intervals = extract_intervals_meditaatio(
        events, 
        raw_task.info['sfreq'], 
        raw_task.first_samp,
        task_cond)
    rest_intervals = extract_intervals_meditaatio_rest(
        raw_rest.times, rest_cond)

    subject = fname.split('_block')[0]
    identifier = fname.split('_tsss')[0]
    trans = os.path.join(folder, subject + '-trans.fif')

    raw_task.filter(*prefilter_band)
    raw_task.resample(sampling_rate_raw)
    raw_task.drop_channels([ch for idx, ch in enumerate(raw_task.info['ch_names'])
                            if idx not in mne.pick_types(raw_task.info, meg=True)])

    raw_rest.filter(*prefilter_band)
    raw_rest.resample(sampling_rate_raw)
    raw_rest.drop_channels([ch for idx, ch in enumerate(raw_rest.info['ch_names'])
                            if idx not in mne.pick_types(raw_rest.info, meg=True)])

    if compute_stc:
        stc_task = create_vol_stc(
            raw=raw_task, 
            trans=trans, 
            subject=subject, 
            noise_cov=noise_cov, 
            spacing=vol_spacing,
            mne_method=mne_method,
            mne_depth=mne_depth,
            subjects_dir=subjects_dir) 

        stc_rest = create_vol_stc(
            raw=raw_rest, 
            trans=trans, 
            subject=subject, 
            noise_cov=noise_cov, 
            spacing=vol_spacing,
            mne_method=mne_method,
            mne_depth=mne_depth,
            subjects_dir=subjects_dir) 

        vertices = stc_task.vertices[0]

    cond_data_task = {}

    print("Prepare using PSD..")

    for key, ivals in task_intervals.items():
        cond_blocks = []
        for ival in ivals:
            start = int(ival[0]*raw_task.info['sfreq'])
            end = int(ival[1]*raw_task.info['sfreq'])
            if compute_stc:
                cond_blocks.append(stc_task.data[:, start:end])
            else:
                cond_blocks.append(raw_task._data[:, start:end])

        concatenated = np.concatenate(cond_blocks, axis=1)

        psd, freqs = mne.time_frequency.psd_array_welch(
            concatenated, 
            sfreq=raw_task.info['sfreq'],
            n_fft=int(raw_task.info['sfreq']*2))

        freqs_idxs = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]

        cond_data_task[key] = np.mean(psd[:, freqs_idxs], axis=1)

    cond_data_rest = {}

    for key, ivals in rest_intervals.items():
        cond_blocks = []
        for ival in ivals:
            start = int(ival[0]*raw_rest.info['sfreq'])
            end = int(ival[1]*raw_rest.info['sfreq'])
            if compute_stc:
                cond_blocks.append(stc_rest.data[:, start:end])
            else:
                cond_blocks.append(raw_rest._data[:, start:end])

        concatenated = np.concatenate(cond_blocks, axis=1)

        psd, freqs = mne.time_frequency.psd_array_welch(
            concatenated, 
            sfreq=raw_rest.info['sfreq'],
            n_fft=int(raw_rest.info['sfreq']*2))

        freqs_idxs = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]

        cond_data_rest[key] = np.mean(psd[:, freqs_idxs], axis=1)

    from collections import OrderedDict
    cond_data = OrderedDict()

    for key in sorted(list(set(list(cond_data_rest.keys()) + list(cond_data_task.keys())))):
        data = []
        if key in cond_data_rest:
            data.append(cond_data_rest[key])
        if key in cond_data_task:
            data.append(cond_data_task[key])

        cond_data[key] = np.mean(data, axis=0)

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
    for key, data in cond_data.items():
        name = identifier + '_' + key
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 15)
        if compute_stc:
            plot_vol_stc_brainmap(data, vertices, vol_spacing, subjects_dir,
                ax) 
        else:
            plot_sensor_topomap(data, raw_task.info, ax)

        if save_path:
            fig.savefig(os.path.join(activation_maps_path, name + '.png'),
                        dpi=50)

    # plot contrasts
    done = []
    for key_1, data_1 in cond_data.items():
        for key_2, data_2 in cond_data.items():
            if (key_1, key_2) in done or (key_2, key_1) in done or key_1 == key_2:
                continue
            done.append((key_1, key_2))

            name = identifier + '_' + key_1 + '_' + key_2

            fig, ax = plt.subplots()
            fig.set_size_inches(20, 15)

            if compute_stc:
                plot_vol_stc_brainmap(data_2 - data_1, vertices, vol_spacing, subjects_dir,
                    ax) 
            else:
                plot_sensor_topomap(data_2 - data_1, raw_task.info, ax)

            if save_path:
                fig.savefig(os.path.join(contrast_maps_path, name + '.png'),
                            dpi=25)

    if save_path:
        for key, data in cond_data.items():
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
        for key_1, data_1 in cond_data.items():
            for key_2, data_2 in cond_data.items():
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

