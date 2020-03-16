PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('font', size=3)
    matplotlib.rc('figure', max_open_warning=0)
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


def plot_topomap_difference(data_1, data_2, info, vmin, vmax):
    data_1 = data_1.copy()
    data_2 = data_2.copy()

    from mne.channels.layout import (_merge_grad_data, find_layout,
                                     _pair_grad_sensors)
    picks, pos = _pair_grad_sensors(info, find_layout(info))
    data_1 = _merge_grad_data(data_1[picks], method='rms').reshape(-1)
    data_2 = _merge_grad_data(data_2[picks], method='rms').reshape(-1)
    
    data = data_2 - data_1

    fig, ax = plt.subplots()
    mne.viz.topomap.plot_topomap(data, pos, axes=ax, vmin=vmin, vmax=vmax)

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    parser.add_argument('--contrast')
    parser.add_argument('--raw', nargs='+')
    parser.add_argument('--empty', nargs='+')
    parser.add_argument('--save-path')

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

    subjects = []
    
    for raw_fname in cli_args.raw:

        path = raw_fname

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
            ['mind', 'plan', 'anx'])

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

        subject_data['name'] = subject
        subject_data['intervals'] = intervals

        print("Prepare using hilbert")
        stc_data = prepare_hilbert(stc.data, sampling_rate_raw, sampling_rate_hilbert)
        subject_data['stc_data'] = stc_data

        sensor_data = prepare_hilbert(raw._data, sampling_rate_raw, sampling_rate_hilbert)
        subject_data['sensor_data'] = sensor_data

        subjects.append(subject_data)

        vertices = stc.vertices

    for subject in subjects:
        task_stc_data = {}
        task_sensor_data = {}
        for key, ivals in subject['intervals'].items():
            if key not in ['mind', 'plan', 'anx']:
                continue

            task_stc_data[key] = []
            task_sensor_data[key] = []

            length = 2
            for ival in ivals:
                subivals = [(istart*sampling_rate_hilbert, (istart + length)*sampling_rate_hilbert) 
                            for istart in range(int(ival[0]), int(ival[1]), length)]
                for subival in subivals:
                    start = int(subival[0])
                    end = int(subival[1])

                    task_stc_data[key].append(np.mean(subject['stc_data'][:, start:end], axis=1))
                    task_sensor_data[key].append(np.mean(subject['sensor_data'][:, start:end], axis=1))

        # find vmin vmax
        vmax_mean = None
        vmax_cbrtmean = None
        for key, data in task_sensor_data.items():
            mean = np.mean(data, axis=0)
            mean_max = np.max(mean)
            cbrtmean_max = np.max(np.cbrt(mean))
            if vmax_mean is None or vmax_mean < mean_max:
                vmax_mean = mean_max
            if vmax_cbrtmean is None or vmax_cbrtmean < cbrtmean_max:
                vmax_cbrtmean = cbrtmean_max

        # every sensor state separately
        for key, data in task_sensor_data.items():
            mean = np.mean(data, axis=0)
            cbrtmean = np.cbrt(mean)

            if save_path:
                topomap_path = os.path.join(save_path, 'topomaps')
                if not os.path.exists(topomap_path):
                    os.makedirs(topomap_path)

            zerov = np.zeros(mean.shape)

            name = subject['name'] + '_' + key + '_mean_sensor'
            fig = plot_topomap_difference(zerov, mean, raw.info, vmin=None, vmax=vmax_mean)
            if save_path:
                fig.savefig(os.path.join(topomap_path, name + '.png'))

            name = subject['name'] + '_' + key + '_cbrtmean_sensor'
            fig = plot_topomap_difference(zerov, cbrtmean, raw.info, vmin=None, vmax=vmax_cbrtmean)
            if save_path:
                fig.savefig(os.path.join(topomap_path, name + '.png'))

        # find vmin and vmax
        vmax_mean = None
        vmax_cbrtmean = None
        for key_1, data_1 in task_sensor_data.items():
            for key_2, data_2 in task_sensor_data.items():
                if key_1 == key_2:
                    continue
                mean_1 = np.mean(data_1)
                mean_2 = np.mean(data_2)
                cbrtmean_1 = np.cbrt(mean_1)
                cbrtmean_2 = np.cbrt(mean_2)
                mean_max = np.max(np.abs(mean_2 - mean_1)) * 4
                cbrtmean_max = np.max(np.abs(cbrtmean_2 - cbrtmean_1)) * 4

                if vmax_mean is None or vmax_mean < mean_max:
                    vmax_mean = mean_max
                if vmax_cbrtmean is None or vmax_cbrtmean < cbrtmean_max:
                    vmax_cbrtmean = cbrtmean_max

        vmin_mean = -vmax_mean
        vmin_cbrtmean = -vmax_cbrtmean

        # sensor contrasts
        done = []
        for key_1, data_1 in task_sensor_data.items():
            for key_2, data_2 in task_sensor_data.items():
                if (key_1, key_2) in done or (key_2, key_1) in done or key_1 == key_2:
                    continue
                done.append((key_1, key_2))
                mean_1 = np.mean(data_1, axis=0)
                mean_2 = np.mean(data_2, axis=0)
                cbrtmean_1 = np.cbrt(mean_1)
                cbrtmean_2 = np.cbrt(mean_2)

                if save_path:
                    topomap_path = os.path.join(save_path, 'contrast_topomaps')
                    if not os.path.exists(topomap_path):
                        os.makedirs(topomap_path)

                name = subject['name'] + '_' + key_1 + '_' + key_2 + '_mean_sensor'
                fig = plot_topomap_difference(mean_1, mean_2, 
                                              raw.info, vmin=vmin_mean, vmax=vmax_mean)
                if save_path:
                    fig.savefig(os.path.join(topomap_path, name + '.png'))

                name = subject['name'] + '_' + key_1 + '_' + key_2 + '_cbrtmean_sensor'
                fig = plot_topomap_difference(cbrtmean_1, cbrtmean_2, 
                                              raw.info, vmin=vmin_cbrtmean, vmax=vmax_cbrtmean)
                if save_path:
                    fig.savefig(os.path.join(topomap_path, name + '.png'))

        # find vmin vmax
        vmax_mean = None
        vmax_cbrtmean = None
        for key, data in task_stc_data.items():
            mean = np.mean(data, axis=0)
            mean_max = np.max(mean)
            cbrtmean_max = np.max(np.cbrt(mean))
            if vmax_mean is None or vmax_mean < mean_max:
                vmax_mean = mean_max
            if vmax_cbrtmean is None or vmax_cbrtmean < cbrtmean_max:
                vmax_cbrtmean = cbrtmean_max

        # every stc state separately
        for key, data in task_stc_data.items():
            mean = np.mean(data, axis=0)
            cbrtmean = np.cbrt(mean)

            name = subject['name'] + '_' + key + '_mean_stc'
            plot_vol_stc_brainmap(save_path, name, 
                mean, vertices, vol_spacing, subjects_dir, vmax=vmax_mean) 

            name = subject['name'] + '_' + key + '_cbrtmean_stc'
            plot_vol_stc_brainmap(save_path, name, 
                cbrtmean, vertices, vol_spacing, subjects_dir, vmax=vmax_cbrtmean) 

        # find vmin and vmax
        vmax_mean = None
        vmax_cbrtmean = None
        for key_1, data_1 in task_stc_data.items():
            for key_2, data_2 in task_stc_data.items():
                if key_1 == key_2:
                    continue
                mean_1 = np.mean(data_1)
                mean_2 = np.mean(data_2)
                cbrtmean_1 = np.cbrt(mean_1)
                cbrtmean_2 = np.cbrt(mean_2)
                mean_max = np.max(np.abs(mean_2 - mean_1))
                cbrtmean_max = np.max(np.abs(cbrtmean_2 - cbrtmean_1))

                if vmax_mean is None or vmax_mean < mean_max:
                    vmax_mean = mean_max
                if vmax_cbrtmean is None or vmax_cbrtmean < cbrtmean_max:
                    vmax_cbrtmean = cbrtmean_max

        # stc contrasts
        done = []
        for key_1, data_1 in task_stc_data.items():
            for key_2, data_2 in task_stc_data.items():
                if (key_1, key_2) in done or (key_2, key_1) in done or key_1 == key_2:
                    continue
                done.append((key_1, key_2))
                mean_1 = np.mean(data_1, axis=0)
                mean_2 = np.mean(data_2, axis=0)
                cbrtmean_1 = np.cbrt(mean_1)
                cbrtmean_2 = np.cbrt(mean_2)
                mean = mean_2 - mean_1
                cbrtmean = cbrtmean_2 - cbrtmean_1

                name = subject['name'] + '_' + key_1 + '_' + key_2 + '_mean_stc'
                plot_vol_stc_brainmap(save_path, name, 
                    mean, vertices, vol_spacing, subjects_dir, folder_name='contrast_vol_brains',
                    vmax=vmax_mean)

                name = subject['name'] + '_' + key_1 + '_' + key_2 + '_cbrtmean_stc'
                plot_vol_stc_brainmap(save_path, name, 
                    cbrtmean, vertices, vol_spacing, subjects_dir, folder_name='contrast_vol_brains',
                    vmax=vmax_cbrtmean) 

        # save stc means to file
        for key, data in task_stc_data.items():
            mean = np.mean(data, axis=0)
            cbrtmean = np.cbrt(mean)
            logmean = np.log(mean)

            if save_path:
                data_path = os.path.join(save_path, 'vol_brains_data')
                if not os.path.exists(data_path):
                    os.makedirs(data_path)

            if save_path:
                name = subject['name'] + '_' + key + '_mean_stc'
                with open(os.path.join(data_path, name + '.csv'), 'w') as f:
                    f.write(', '.join([str(elem) for elem in vertices.tolist()]) + '\n')
                    f.write(', '.join([str(elem) for elem in mean.tolist()]))

                name = subject['name'] + '_' + key + '_cbrtmean_stc'
                with open(os.path.join(data_path, name + '.csv'), 'w') as f:
                    f.write(', '.join([str(elem) for elem in vertices.tolist()]) + '\n')
                    f.write(', '.join([str(elem) for elem in cbrtmean.tolist()]))

                name = subject['name'] + '_' + key + '_logmean_stc'
                with open(os.path.join(data_path, name + '.csv'), 'w') as f:
                    f.write(', '.join([str(elem) for elem in vertices.tolist()]) + '\n')
                    f.write(', '.join([str(elem) for elem in logmean.tolist()]))

        # save stc contrasts to file
        done = []
        for key_1, data_1 in task_stc_data.items():
            for key_2, data_2 in task_stc_data.items():
                if (key_1, key_2) in done or (key_2, key_1) in done or key_1 == key_2:
                    continue
                done.append((key_1, key_2))
                mean_1 = np.mean(data_1, axis=0)
                mean_2 = np.mean(data_2, axis=0)
                cbrtmean_1 = np.cbrt(mean_1)
                cbrtmean_2 = np.cbrt(mean_2)
                logmean_1 = np.log(mean_1)
                logmean_2 = np.log(mean_2)
         
                mean = mean_2 - mean_1
                cbrtmean = cbrtmean_2 - cbrtmean_1
                logmean = logmean_2 - logmean_1

                if save_path:
                    data_path = os.path.join(save_path, 'contrast_vol_brains_data')
                    if not os.path.exists(data_path):
                        os.makedirs(data_path)

                if save_path:
                    name = subject['name'] + '_' + key_1 + '_' + key_2 + '_mean_stc'
                    with open(os.path.join(data_path, name + '.csv'), 'w') as f:
                        f.write(', '.join([str(elem) for elem in vertices.tolist()]) + '\n')
                        f.write(', '.join([str(elem) for elem in mean.tolist()]))

                    name = subject['name'] + '_' + key_1 + '_' + key_2 + '_cbrtmean_stc'
                    with open(os.path.join(data_path, name + '.csv'), 'w') as f:
                        f.write(', '.join([str(elem) for elem in vertices.tolist()]) + '\n')
                        f.write(', '.join([str(elem) for elem in cbrtmean.tolist()]))

                    name = subject['name'] + '_' + key_1 + '_' + key_2 + '_logmean_stc'
                    with open(os.path.join(data_path, name + '.csv'), 'w') as f:
                        f.write(', '.join([str(elem) for elem in vertices.tolist()]) + '\n')
                        f.write(', '.join([str(elem) for elem in logmean.tolist()]))

    import pdb; pdb.set_trace()
    print("miau")
