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
import time

from collections import OrderedDict

import nibabel as nib
import mne
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import sklearn
import hmmlearn

from nilearn.plotting import plot_glass_brain

from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

from scipy.signal import hilbert
from scipy.signal import decimate

# from icasso import Icasso

from signals.cibr.common import preprocess


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

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    mne_method, mne_depth = 'dSPM', None
    band = (7, 14)
    sampling_rate_raw = 50.0
    tasks = ['mind', 'plan', 'anx']
    
    surf_spacing = 'ico3'
    vol_spacing = '10'
    bootstrap_iterations = 100
    page = 4

    # hilbert ica settings
    sampling_rate_hilbert = 1.0

    current_time = 0
    subject_data = []
    for path_idx, path in enumerate(cli_args.raws):
        folder = os.path.dirname(path)
        fname = os.path.basename(path)

        # this is for meditation
        subject = '_'.join(fname.split('_')[:2])

        print("Handling " + path)

        raw = mne.io.Raw(path, preload=True)
        raw.resample(sampling_rate_raw)
        raw, events = preprocess(raw, filter_=band, min_duration=1)

        intervals = extract_intervals_meditation(
            events, 
            raw.info['sfreq'], 
            raw.first_samp,
            tasks)

        subject_item = {}
        subject_item['name'] = subject
        subject_item['intervals'] = intervals
        subject_item['start'] = current_time

        def prepare_hilbert(data):
            # get envelope as abs of analytic signal
            import time
            start = time.time()
            rowsplits = np.array_split(data, 4, axis=0)
            env_rowsplits = []
            for rowsplit in rowsplits:
                blocks = np.array_split(rowsplit, 2, axis=1)
                print("Compute hilbert for each block:")
                env_blocks = []
                for block in blocks:
                    print("Block shape: " + str(block.shape))
                    env_blocks.append(np.abs(hilbert(block)))
                    print("Elapsed: " + str(time.time() - start))

                env_rowsplits.append(np.concatenate(env_blocks, axis=1))
            env = np.concatenate(env_rowsplits, axis=0)

            print("Decimating")
            # decimate first with five
            decimated = decimate(env, 5)
            # and then the rest
            factor = int(((sampling_rate_raw / 5.0) / sampling_rate_hilbert))
            decimated = decimate(decimated, factor)
            return decimated

        print("Original data shape: " + str(raw._data.shape)) 
        print("Prepare using hilbert")
        data = prepare_hilbert(raw._data)

        # CROP TO MEDITATIONS
        data_array = []
        for key, ivals in intervals.items():
            if key == 'mind':
                for ival in ivals:
                    start_idx = int((ival[0]) * sampling_rate_hilbert)
                    end_idx = int((ival[1]) * sampling_rate_hilbert)
                    data_array.append(data[:, start_idx:end_idx])

        data = np.concatenate(data_array, axis=-1)

        # zscore
        data = (data - np.mean(data)) / np.std(data)

        subject_item['data'] = data

        current_time += data.shape[-1] / sampling_rate_hilbert

        subject_data.append(subject_item)

    brain_data = np.concatenate([subject['data'] for subject in subject_data], 
                                axis=-1)

    brain_pca = sklearn.decomposition.PCA(n_components=20, whiten=True)

    brain_data_wh = brain_pca.fit_transform(brain_data.T).T

    from hmmlearn.hmm import GaussianHMM
    markov = GaussianHMM(
        n_components=6,
        covariance_type='full')

    markov.fit(brain_data_wh.T)

    # plot pure maps
    # for comp_idx in range(brain_pca.components_.shape[0]):
    #     component = brain_pca.components_[comp_idx]
    #     plot_vol_stc_brainmap(save_path, 'pure_' + str(comp_idx+1).zfill(2), component, 
    #                           vertices, vol_spacing, subjects_dir)

    # plot state maps
    # for state_idx in range(markov.means_.shape[0]):
    #     weights = markov.means_[state_idx]
    #     print("For state " + str(state_idx+1) + " the weights are: ")
    #     print(str(weights))
    #     statemap = np.dot(weights, brain_pca.components_)
    #     plot_vol_stc_brainmap(save_path, 'state_' + str(state_idx+1).zfill(2), statemap, 
    #                           vertices, vol_spacing, subjects_dir)

    # plot time series
    state_chain = markov.predict(brain_data_wh.T)

    nrows = len(subject_data)
    colors = plt.cm.get_cmap('gist_rainbow', markov.means_.shape[0])

    fig, axes = plt.subplots(ncols=1, nrows=nrows)
    for row_idx in range(nrows):
        start = int(subject_data[row_idx]['start'] * float(sampling_rate_hilbert))
        if row_idx != nrows - 1:
            end = int(subject_data[row_idx+1]['start'] * float(sampling_rate_hilbert))
        else:
            end = state_chain.shape[0]

        if nrows > 1:
            ax = axes[row_idx]
        else:
            ax = axes

        # ax.plot(state_chain[start:end])
        for idx in range(start, end-1):
            ax.axvspan(idx, idx+1, alpha=0.5, 
                       color=colors(state_chain[idx]))

        patches = []
        for state_idx in range(markov.means_.shape[0]):
            patches.append(mpatches.Patch(color=colors(state_idx), 
                                          label=('State ' + str(state_idx+1))))
        ax.legend(handles=patches)
    
    if save_path:
        series_path = os.path.join(save_path, 'series')
        if not os.path.exists(series_path):
            os.makedirs(series_path)

        name = 'state_ts'
        path = os.path.join(series_path, name + '.png')

        fig.savefig(path, dpi=620)

    import pdb; pdb.set_trace()
    print("miau, lets look at the summaries")

