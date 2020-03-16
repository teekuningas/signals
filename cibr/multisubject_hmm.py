PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('figure', max_open_warning=0)
    matplotlib.use('Agg')
    matplotlib.rc('font', size=3)

import pyface.qt

import sys
import gc
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
from statsmodels.stats.anova import AnovaRM

from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM

from scipy.signal import hilbert
from scipy.signal import decimate
import scipy.fftpack as fftpack

from signals.cibr.lib.stc import create_vol_stc
from signals.cibr.lib.stc import plot_vol_stc_brainmap

from signals.cibr.lib.hmm import fractional_occupancy

from signals.cibr.lib.hmm import plot_state_series
from signals.cibr.lib.hmm import plot_task_comparison

from signals.cibr.lib.triggers import extract_intervals_fdmsa_ic
from signals.cibr.lib.triggers import extract_intervals_hengitys
from signals.cibr.lib.triggers import extract_intervals_multimodal
from signals.cibr.lib.triggers import extract_intervals_fdmsa_rest
from signals.cibr.lib.triggers import extract_intervals_meditaatio

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def smooth(data, window_len, window='hanning'):
    """ 
    """
    result = []
    for idx in range(data.shape[0]):
        x = data[idx]
        
        s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]

        if window == 'flat': 
            w=np.ones(window_len,'d')
        else:
            w=eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')

        # slice output to match input size
        result.append(y[int((window_len/2-1)):-int(window_len/2)])
    return np.array(result)


def fast_hilbert(x):
    return hilbert(x, fftpack.next_fast_len(x.shape[-1]))[..., :x.shape[-1]]


def prepare_hilbert(data, sfreq, smoothing_window):
    # get envelope as abs of analytic signal
    rowsplits = np.array_split(data, 2, axis=0)
    env_rowsplits = []
    for rowsplit in rowsplits:
        blocks = np.array_split(rowsplit, 4, axis=1)
        env_blocks = []
        for block in blocks:
            env_blocks.append(np.abs(fast_hilbert(block)))

        env_rowsplits.append(np.concatenate(env_blocks, axis=1))
    env = np.concatenate(env_rowsplits, axis=0)

    # check window length before and after
    print("Env shape before " + str(env.shape))
    
    env = smooth(env, int(smoothing_window*sfreq))
    print("Env shape after " + str(env.shape))

    return env


def plot_sensor_topomap(data, info, ax, factor=1.0):
    """
    """
    data = data.copy()

    from mne.channels.layout import (_merge_grad_data, find_layout,
                                     _pair_grad_sensors)
    picks, pos = _pair_grad_sensors(info, find_layout(info))
    data = _merge_grad_data(data[picks], method='rms').reshape(-1)

    # data[data < np.percentile(data, 75)] = np.min(data)
    # vmax = np.max(data)
    # vmin = np.min(data)

    if np.max(data) >= 0:
        pos_limit = np.percentile(data[data >= 0], 75)
        data[(data >= 0) & (data < pos_limit)] = 0
        data[(data >= 0) & (data >= pos_limit)] -= pos_limit
    if np.min(data) <= 0:
        neg_limit = np.percentile(data[data <= 0], 25)
        data[(data <= 0) & (data > neg_limit)] = 0
        data[(data <= 0) & (data <= neg_limit)] -= neg_limit

    vmax = np.max(np.abs(data)) / factor
    vmin = -vmax
    
    mne.viz.topomap.plot_topomap(data, pos, axes=ax, vmin=vmin, vmax=vmax,
                                 cmap='RdBu_r')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', nargs='+')
    parser.add_argument('--empty', nargs='+')
    parser.add_argument('--save-path')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    vol_spacing = '10'
    mne_method, mne_depth = 'dSPM', None

    compute_stc = True

    sampling_rate_raw = 100.0
    prefilter_band = (1, 40)
    decimation_factor = 10
    decimate_plot_to = 2.0

    bands = [
        ('alpha', (7, 14)), 
    #     ('beta', (18, 32)), 
    ]
    smoothing_window = 0.1  # in s

    n_pca_components = 30
    n_states = 6

    evoked_baseline = (-2.5, -0.5)
    evoked_times = (-3.0, 4.0)

    timeblocks = [
        ('full', (0, 1800)), 
    ]

    cmap = plt.cm.get_cmap('gist_rainbow', n_states)
    state_colors = [cmap(idx) for idx in range(n_states)]

    if compute_stc:
        # process similarly to input data 
        empty_paths = cli_args.empty
        empty_raws = []
        for fname in empty_paths:
            raw = mne.io.Raw(fname, preload=True, verbose='error')
            raw.filter(*prefilter_band)
            raw.crop(tmin=(raw.times[0]+3), tmax=(raw.times[-1]-3))
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

    subjects = []
    data = []
    
    for raw_fname in cli_args.raw:

        path = raw_fname
        folder = os.path.dirname(path)
        fname = os.path.basename(path)

        print("Handling " + path)

        raw = mne.io.Raw(path, preload=True, verbose='error')

        # find triggers before resampling so that no information is lost
        events = mne.find_events(raw, shortest_event=1, min_duration=1/raw.info['sfreq'], 
                                 uint_cast=True, verbose='warning')

        # fdmsa intervals
        # intervals = extract_intervals_fdmsa_ic(events, raw.info['sfreq'],
        #                                        raw.first_samp)
        # code = fname.split('_tsss')[0].split('IC')[-1][2:]
        # subject = 'FDMSA_' + code
        # identifier = fname.split('_tsss')[0] 
        # trans = os.path.join(folder, subject + '-trans.fif')

        # meditaatio intervals
        intervals = extract_intervals_meditaatio(events, raw.info['sfreq'],
                                                 raw.first_samp, 
                                                 ['mind', 'plan'])
        subject = fname.split('_block')[0]
        identifier = fname.split('_tsss')[0]
        trans = os.path.join(folder, subject + '-trans.fif')

        # multimodal "intervals"
        # intervals = extract_intervals_multimodal(events, raw.info['sfreq'], raw.first_samp)
        # identifier = 'multimodal'

        # fdmsa rest intervals
        # intervals = extract_intervals_fdmsa_rest(events, raw.info['sfreq'], raw.first_samp, raw.times[-1])
        # code = fname.split('_tsss')[0].split('restpre')[-1][1:]
        # subject = 'FDMSA_' + code
        # identifier = fname.split('_tsss')[0] 
        # trans = os.path.join(folder, subject + '-trans.fif')

        # hengitys "intervals"
        # intervals = extract_intervals_hengitys(events, raw.info['sfreq'], raw.first_samp)
        # identifier = 'hengitys'

        raw.filter(*prefilter_band)
        raw.resample(sampling_rate_raw)
        raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names']) 
                           if idx not in mne.pick_types(raw.info, meg=True)])

        sfreq = raw.info['sfreq']

        subject_info = {}
        subject_info['name'] = identifier
        subject_info['intervals'] = intervals
        subjects.append(subject_info)

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

        data_by_bands = []

        for band_name, band in bands:
            print("Prepare " + band_name + " band envelope using hilbert transform.")
            if compute_stc:
                # filter to band
                band_data = mne.filter.filter_data(stc.data.copy(), sfreq, l_freq=band[0], h_freq=band[1])

                # decimate
                if decimation_factor > 1:
                    band_data = np.array([decimate(row, decimation_factor) for row in band_data])

                # compute envelope
                data_by_bands.append((band_name, prepare_hilbert(band_data, sfreq, smoothing_window)))

            else:
                # filter to band
                band_data = mne.filter.filter_data(raw._data.copy(), sfreq, l_freq=band[0], h_freq=band[1])

                # decimate
                if decimation_factor > 1:
                    band_data = np.array([decimate(row, decimation_factor) for row in band_data])

                # compute envelope
                data_by_bands.append((band_name, prepare_hilbert(band_data, sfreq, smoothing_window)))

        if decimation_factor > 1:
            sfreq = sfreq / decimation_factor

        data.append(data_by_bands)

    concatenated_data = []

    current_time = 0
    for idx, subject in enumerate(subjects):
        sub_data = []
        for key, band_data in data[idx]:
            band_data = band_data / np.mean(band_data, axis=1)[:, np.newaxis]
            sub_data.append(band_data)
        sub_data = np.concatenate(sub_data, axis=0)
        concatenated_data.append(sub_data)

        subject['start_time'] = current_time
        current_time += sub_data.shape[1] / sfreq
    
    data = None; gc.collect()
    data = np.concatenate(concatenated_data, axis=1)
    concatenated_data = None, gc.collect()

    pca = PCA(n_components=n_pca_components, whiten=True)
    comps = pca.fit_transform(np.array(data.T)).T

    # plot pca comps
    for comp_idx in range(n_pca_components):
        print("Stats for comp " + str(comp_idx+1))

        comps_by_band = np.array_split(pca.components_[comp_idx], len(bands))
        fig, axes = plt.subplots(ncols=1, nrows=len(bands))

        vmax = 0
        for band_idx in range(len(bands)):
            if np.max(np.abs(comps_by_band[band_idx])) > vmax:
                vmax = np.max(np.abs(comps_by_band[band_idx]))

        for band_idx in range(len(bands)):
            print("Band " + str(bands[band_idx][0]) + ": " + str(np.max(np.abs(comps_by_band[band_idx]))))

            if len(bands) > 1:
                ax = axes[band_idx]
            else:
                ax = axes

            if compute_stc:
                plot_vol_stc_brainmap(comps_by_band[band_idx],
                                      vertices, vol_spacing, 
                                      subjects_dir, ax, vmax=vmax, cap=0.8)
            else:
                plot_sensor_topomap(comps_by_band[band_idx], raw.info, ax, 
                                    factor=np.max(np.abs(comps_by_band[band_idx]))/vmax)

            ax.set_title(str(bands[band_idx][0]))

        if save_path:
            comp_path = os.path.join(save_path, 'comps')
            if not os.path.exists(comp_path):
                os.makedirs(comp_path)

            name = 'comp_' + str(comp_idx+1).zfill(2)
            path = os.path.join(comp_path, name + '.png')
            fig.savefig(path, dpi=310)

    markov = GaussianHMM(
        n_components=n_states,
        covariance_type='full',
        verbose=True,
        n_iter=10000)

    print("Fitting HMM")
    markov.fit(comps.T)

    for state_idx in range(n_states):
        weights = markov.means_[state_idx]
        backproj = np.dot(weights, pca.components_) + pca.mean_
        statemaps = np.array_split(backproj, len(bands))
        fig, axes = plt.subplots(ncols=1, nrows=len(bands))
        
        vmax = 0
        for band_idx in range(len(bands)):
            if np.max(np.abs(statemaps[band_idx])) > vmax:
                vmax = np.max(np.abs(statemaps[band_idx]))

        for band_idx in range(len(bands)):

            if len(bands) > 1:
                ax = axes[band_idx]
            else:
                ax = axes

            if compute_stc:
                plot_vol_stc_brainmap(statemaps[band_idx],
                                      vertices, vol_spacing, 
                                      subjects_dir, ax, vmax=vmax, cap=0.8)
            else:
                plot_sensor_topomap(statemaps[band_idx], raw.info, ax,
                                    factor=np.max(np.abs(statemaps[band_idx]))/vmax)

            ax.set_title(str(bands[band_idx][0]))

        if save_path:
            state_path = os.path.join(save_path, 'states')
            if not os.path.exists(state_path):
                os.makedirs(state_path)

            name = 'state_' + str(state_idx+1).zfill(2)
            path = os.path.join(state_path, name + '.png')
            fig.savefig(path, dpi=310)


    print("Inferring hidden states")
    import pdb; pdb.set_trace()
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
    cmap = plt.cm.get_cmap('brg', len(subject['intervals']))
    for subject in subjects:
        subject_annot = []
        for idx, (key, ivals) in enumerate(subject['intervals'].items()):
            # fdmsa:
            if key == 'heart':
                subject_annot.append((key, 'red', ivals))
            elif key == 'note':
                subject_annot.append((key, 'blue', ivals))
            else:
                subject_annot.append((key, cmap(idx), ivals))
        task_annotations.append(subject_annot)

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
                                    state_colors=state_colors,
                                    task_annotations=annotations,
                                    probabilistic=True,
                                    decimate_to=decimate_plot_to)

            series_path = os.path.join(save_path, 'series')
            if not os.path.exists(series_path):
                os.makedirs(series_path)

            name = timeblock_name + '_subs' + str(block_idx+1).zfill(2)
            path = os.path.join(series_path, name + '.png')
            fig.savefig(path, dpi=620)

    # statistical analysis

    columns = ['subject', 'occ_freq', 'condition', 'state']

    data = []
    for state_idx in range(n_states):
        for subject_idx, subject in enumerate(subjects):
            state_chain = state_chain_by_subject[subject_idx]
            for ival_key, ivals in subject['intervals'].items():
                ival_chain = []
                for ival in ivals:
                    start_idx = int(ival[0] * sfreq)
                    end_idx = int(ival[1] * sfreq)
                    ival_chain.append(state_chain[start_idx:end_idx])
                ival_chain = np.concatenate(ival_chain, axis=0)
                data.append([subject['name'], 
                             fractional_occupancy(ival_chain, sfreq)[state_idx], 
                             ival_key,
                             str(state_idx).zfill(2)])

    df = pd.DataFrame(data, columns=columns)
    for state_idx in range(n_states):
        # select state specific rows
        print("Anova for state " + str(state_idx+1))

        state_rows = df[df['state'] == str(state_idx).zfill(2)]
        anova = AnovaRM(state_rows, 'occ_freq', 'subject', 
                        within=['condition'])
        res = anova.fit()
        print(res)

    # do group comparisons later here..


    # plot chain stats
    for subject_idx, subject in enumerate(subjects):

        state_chain = state_chain_by_subject[subject_idx]

        # plot fractional occupancies
        title = "Frac occ of " + subject['name']
        ylabel = 'Fractional occupancy'
        occ_fig = plot_task_comparison(state_chain, sfreq, subject['intervals'].items(),
                                       fractional_occupancy, ylabel=ylabel, title=title)

        if save_path:
            chain_stats_path = os.path.join(save_path, 'chain_stats')
            if not os.path.exists(chain_stats_path):
                os.makedirs(chain_stats_path)

            fname = 'frac_occ_' + subject['name']
            path = os.path.join(chain_stats_path, fname + '.png')
            occ_fig.savefig(path, dpi=155)

    # evoked fractional occupancies
    for subject_idx, subject in enumerate(subjects):
        state_chain = state_chain_by_subject[subject_idx]
        posterior_epochs = {}
        for key, ivals in intervals.items():
            for ival in ivals:
                if key not in posterior_epochs:
                    posterior_epochs[key] = []

                start = int((ival[0] + evoked_times[0]) * sfreq)
                end = int((ival[0] + evoked_times[1]) * sfreq)
                baseline_start = int((ival[0] + evoked_baseline[0]) * sfreq)
                baseline_end = int((ival[0] + evoked_baseline[1]) * sfreq)

                if end > state_chain.shape[0]:
                    continue

                seg_length = int((evoked_times[1] - evoked_times[0]) * sfreq)

                epochs = (state_chain[start:start+seg_length, :] - 
                          np.mean(state_chain[baseline_start:baseline_end, :], axis=0))

                posterior_epochs[key].append(epochs)

        # plot evoked
        for key, epochs in posterior_epochs.items():
            fig, ax = plt.subplots()

            evoked_data = np.mean(epochs, axis=0)

            evokeds = OrderedDict()
            for state_idx, evoked in enumerate(evoked_data.T):
                info = mne.create_info(
                    ['State'], sfreq)
                evoked_array = mne.EvokedArray(evoked[np.newaxis, :],
                                               info, tmin=evoked_times[0])
                evoked_array.comment = str(state_idx+1).zfill(2)
                evokeds[evoked_array.comment] = evoked_array

            evok_idx = 0
            def ci(*args):
                global evok_idx

                relev_epochs = np.array(epochs)[:, :, evok_idx]

                evok_idx += 1

                from mne.stats.permutations import _ci
                ci = _ci(relev_epochs, ci=.99)
                for tidx in range(ci.shape[1]):
                    if ci[0, tidx] < 0 and ci[1, tidx] > 0:
                        ci[:, tidx] = 0

                return ci

            # construct evoked array
            title = subject['name'] + '_' + '_'.join(key.split(' '))
            mne.viz.plot_compare_evokeds(evokeds, axes=ax, picks=[0], title=title, vlines=[0],
                                         colors=state_colors, ci=ci)

            if save_path:
                evoked_occ_path = os.path.join(save_path, 'evoked_occ')
                if not os.path.exists(evoked_occ_path):
                    os.makedirs(evoked_occ_path)

                fname = title
                path = os.path.join(evoked_occ_path, fname + '.png')
                fig.savefig(path, dpi=155)


