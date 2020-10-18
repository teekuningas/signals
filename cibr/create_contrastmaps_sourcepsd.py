PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('figure', max_open_warning=0)
    matplotlib.rc('font', size=15)
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
from signals.cibr.lib.triggers import extract_intervals_fdmsa_ic

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw')
    parser.add_argument('--empty', nargs='+')
    parser.add_argument('--save_path')
    parser.add_argument('--tasks')
    parser.add_argument('--band')
    parser.add_argument('--compute_stc')
    parser.add_argument('--mne_method')
    parser.add_argument('--depth')
    parser.add_argument('--spacing')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    mne_method = 'dSPM'
    if cli_args.mne_method is not None:
        mne_method = str(cli_args.mne_method)

    mne_depth = None
    if cli_args.depth is not None:
        mne_depth = float(cli_args.depth)

    vol_spacing = '10'
    if cli_args.spacing is not None:
        vol_spacing = str(cli_args.spacing)

    sampling_rate_raw = 100.0
    prefilter_band = (1, 40)

    compute_stc = True
    if cli_args.compute_stc:
        compute_stc = True if cli_args.compute_stc == 'true' else False

    # band = (17, 25)
    band = (7, 14)
    if cli_args.band:
        band = (int(cli_args.band.split()[0]), int(cli_args.band.split()[1]))

    tasks = ('mind', 'plan')
    if cli_args.tasks:
        tasks = cli_args.tasks.split()[0], cli_args.tasks.split()[1]

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

    # intervals = extract_intervals_fdmsa_ic(
    #     events,
    #     raw.info['sfreq'],
    #     raw.first_samp)
    # code = fname.split('_tsss')[0].split('IC')[-1][2:]
    # subject = 'FDMSA_' + code
    # identifier = subject
    # trans = os.path.join(folder, subject + '-trans.fif')

    raw.filter(*prefilter_band)
    raw.resample(sampling_rate_raw)
    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names'])
                       if idx not in mne.pick_types(raw.info, meg=True)])


    # if compute_stc:
        # stc = create_vol_stc(
        #     raw=raw, 
        #     trans=trans, 
        #     subject=subject, 
        #     noise_cov=noise_cov, 
        #     spacing=vol_spacing,
        #     mne_method=mne_method,
        #     mne_depth=mne_depth,
        #     subjects_dir=subjects_dir) 

        # vertices = stc.vertices[0]

    task_data = {}
    task_data_psds = {}

    print("Prepare using PSD..")

    for key, ivals in intervals.items():
        task_blocks = []
        for ival in ivals:
            start = int(ival[0]*raw.info['sfreq'])
            end = int(ival[1]*raw.info['sfreq'])

            task_blocks.append(raw._data[:, start:end])
        
        concatenated = np.concatenate(task_blocks, axis=1)

        if compute_stc:
            bem = os.path.join(subjects_dir, subject, 'bem',
                               subject+'-inner_skull-bem-sol.fif')
            src_fname = os.path.join(subjects_dir, subject, 'bem',
                                     subject + '-vol-' + vol_spacing + '-src.fif')
            src = mne.source_space.read_source_spaces(src_fname)

            print("Creating forward solution..")
            fwd = mne.make_forward_solution(
                info=raw.info,
                trans=trans,
                src=src,
                bem=bem,
                meg=True,
                eeg=False,
                verbose='warning')

            print("Creating inverse operator..")
            inv = mne.minimum_norm.make_inverse_operator(
                info=raw.info,
                forward=fwd,
                noise_cov=noise_cov,
                depth=mne_depth,
                fixed=False,
                verbose='warning')

            raw_array = mne.io.RawArray(concatenated, raw.info, first_samp=raw.first_samp, copy='auto')

            stc_psd = mne.minimum_norm.compute_source_psd(
                raw_array, inv, method='dSPM', n_fft=int(raw.info['sfreq']*4), pick_ori=None, dB=False)

            freqs = stc_psd.times
            psd = stc_psd.data
            vertices = stc_psd.vertices[0]

            freqs_idxs = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]

            task_data_psds[key] = psd
            task_data[key] = np.mean(psd[:, freqs_idxs], axis=1)

        else:
            
            psd, freqs = mne.time_frequency.psd_array_welch(
                concatenated, 
                sfreq=raw.info['sfreq'],
                n_overlap=int(raw.info['sfreq']*2 / 2),
                n_fft=int(raw.info['sfreq']*2))

            freqs_idxs = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]

            task_data_psds[key] = psd

            task_data[key] = np.mean(psd[:, freqs_idxs], axis=1)


    if save_path:
        activation_maps_path = os.path.join(save_path, 'activation_maps')
        contrast_maps_path = os.path.join(save_path, 'contrast_maps')
        activation_data_path = os.path.join(save_path, 'activation_data')
        contrast_data_path = os.path.join(save_path, 'contrast_data')
        construction_path = os.path.join(save_path, 'construction')

        try:
            os.makedirs(activation_maps_path)
            os.makedirs(contrast_maps_path)
            os.makedirs(activation_data_path)
            os.makedirs(contrast_data_path)
            os.makedirs(construction_path)
        except FileExistsError:
            pass

    selected_idx = 3600
    task1_label = 'FA' if tasks[0] == 'mind' else ''
    task2_label = 'AT' if tasks[1] == 'anx' else ''

    # fig, ax = plt.subplots()
    # fig.set_size_inches(20, 10)

    # ax.plot(stc.times[:200], stc.data[selected_idx][:200], zorder=1)

    # for ival in intervals[tasks[0]]:
    #     ax.axvspan(xmin=ival[0], xmax=ival[1], facecolor='blue', alpha=0.3, zorder=2)
    # for ival in intervals[tasks[1]]:
    #     ax.axvspan(xmin=ival[0], xmax=ival[1], facecolor='orange', alpha=0.3, zorder=2)

    # from matplotlib.lines import Line2D
    # lines = [
    #     Line2D([0], [0], color='blue', lw=10),
    #     Line2D([0], [0], color='orange', lw=10)
    # ]

    # ax.legend(lines, [task1_label, task2_label])

    # ax.set_ylim(0, 1.5*np.max(stc.data))

    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Noise-normalized MNE current (AU)')
    # if save_path:
    #     fig.savefig(os.path.join(construction_path, 'series' + str(selected_idx).zfill(4) + '.png'),
    #                 dpi=100)

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)

    ax.plot(freqs[1:], task_data_psds[tasks[0]][selected_idx][1:], label=task1_label,
            color='blue')
    ax.plot(freqs[1:], task_data_psds[tasks[1]][selected_idx][1:], label=task2_label,
            color='orange')
    ax.axvspan(xmin=7, xmax=14, facecolor='g', alpha=0.3)
    ax.legend()
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (AU)')
    if save_path:
        fig.savefig(os.path.join(construction_path, 'spectrum' + str(selected_idx).zfill(4) + '.png'),
                    dpi=100)

    # plot every state separately
    for key, data in task_data.items():
        name = identifier + '_' + key
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 15)
        if compute_stc:
            plot_vol_stc_brainmap(data, vertices, vol_spacing, subjects_dir,
                ax, cap=0.5) 
        else:
            plot_sensor_topomap(data, raw.info, ax)

        if save_path:
            fig.savefig(os.path.join(activation_maps_path, name + '.png'),
                        dpi=50)

    # plot contrasts
    done = []
    for key_1, data_1 in task_data.items():
        for key_2, data_2 in task_data.items():
            if (key_1, key_2) in done or (key_2, key_1) in done or key_1 == key_2:
                continue
            done.append((key_1, key_2))

            name = identifier + '_' + key_1 + '_' + key_2

            fig, ax = plt.subplots()
            fig.set_size_inches(20, 15)

            if compute_stc:
                plot_vol_stc_brainmap(data_2 - data_1, vertices, vol_spacing, subjects_dir,
                    ax, cap=0.5) 
            else:
                plot_sensor_topomap(data_2 - data_1, raw.info, ax)

            if save_path:
                fig.savefig(os.path.join(contrast_maps_path, name + '.png'),
                            dpi=25)

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

