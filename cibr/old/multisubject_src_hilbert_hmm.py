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


def create_vol_stc(raw, trans, subject, noise_cov, spacing, 
                   mne_method, mne_depth, subjects_dir):
    """
    """
    
    bem = os.path.join(subjects_dir, subject, 'bem',
                       subject+'-inner_skull-bem-sol.fif')

    src_fname = os.path.join(subjects_dir, subject, 'bem',
                             subject + '-vol-' + spacing + '-src.fif')

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
        limit_depth_chs=False,
        verbose='warning')

    print("Applying inverse operator..")
    stc = mne.minimum_norm.apply_inverse_raw(
        raw=raw,
        inverse_operator=inv,
        lambda2=0.1,
        method=mne_method,
        label=None,
        pick_ori=None,
        verbose='warning')

    return stc

def create_vol_stc_labels(raw, trans, subject, noise_cov, spacing, 
                          mne_method, mne_depth, subjects_dir):
    """
    """
    bem = os.path.join(subjects_dir, subject, 'bem',
                       subject+'-inner_skull-bem-sol.fif')

    src_fname = os.path.join(subjects_dir, subject, 'bem',
                             subject + '-vol-' + spacing + 
                             '-labels-src.fif')

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
        limit_depth_chs=False,
        verbose='warning')

    print("Applying inverse operator..")
    stc = mne.minimum_norm.apply_inverse_raw(
        raw=raw,
        inverse_operator=inv,
        lambda2=0.1,
        method=mne_method,
        label=None,
        pick_ori=None,
        verbose='warning')

    return stc

def plot_vol_stc_brainmap(save_path, name, brainmap, vertices, spacing, subjects_dir):

    fig_ = plt.figure()

    brainmap_inc = brainmap.copy()
    brainmap_dec = brainmap.copy()

    brainmap_inc[brainmap_inc < 0] = 0
    brainmap_dec[brainmap_dec > 0] = 0
    brainmap_dec = -brainmap_dec

    brainmap_inc = (brainmap_inc - np.mean(brainmap_inc)) / np.std(brainmap_inc)
    brainmap_dec = (brainmap_dec - np.mean(brainmap_dec)) / np.std(brainmap_dec)

    stc_inc = mne.source_estimate.VolSourceEstimate(
        brainmap_inc[:, np.newaxis],
        vertices,
        tstep=0.1,
        tmin=0,
        subject='fsaverage')
    stc_dec = mne.source_estimate.VolSourceEstimate(
        brainmap_dec[:, np.newaxis],
        vertices,
        tstep=0.1,
        tmin=0,
        subject='fsaverage')

    src_fname = os.path.join(subjects_dir, 'fsaverage', 'bem',
                             'fsaverage-vol-' + spacing + '-src.fif')
    src = mne.source_space.read_source_spaces(src_fname)

    aseg_fname = os.path.join(subjects_dir, 'fsaverage', 'mri', 'aseg.mgz')
    t1_img = nib.load(aseg_fname)

    nifti_inc = stc_inc.as_volume(src).slicer[:, :, :, 0]
    nifti_dec = stc_dec.as_volume(src).slicer[:, :, :, 0]

    display = plot_glass_brain(t1_img, figure=fig_, display_mode='lyrz')
    display.add_overlay(nifti_inc, alpha=0.9, cmap='Reds')
    display.add_overlay(nifti_dec, alpha=0.5, cmap='Blues')

    if not save_path:
        plt.show()

    if save_path:

        brain_path = os.path.join(save_path, 'vol_brains')
        if not os.path.exists(brain_path):
            os.makedirs(brain_path)

        path = os.path.join(brain_path,
            name + '.png')

        fig_.savefig(path, dpi=620)


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


def prepare_hilbert(data, sampling_rate_raw,
                    sampling_rate_hilbert):
    # get envelope as abs of analytic signal
    import time
    start = time.time()
    rowsplits = np.array_split(data, 2, axis=0)
    env_rowsplits = []
    for rowsplit in rowsplits:
        blocks = np.array_split(rowsplit, 10, axis=1)
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

    dr_method = 'pca'
    mne_method, mne_depth = 'dSPM', None
    band = (15, 25)
    sampling_rate_raw = 100.0
    tasks = ['mind', 'plan', 'anx']
    
    surf_spacing = 'ico3'
    vol_spacing = '10'
    bootstrap_iterations = 100
    page = 4

    # hilbert ica settings
    sampling_rate_hilbert = 1.0

    # process similarly to input data 
    empty_paths = cli_args.empty
    empty_raws = []
    for fname in empty_paths:
        raw = mne.io.Raw(fname, preload=True)
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
    subject_data = []
    for path_idx, path in enumerate(cli_args.raws):
        folder = os.path.dirname(path)
        fname = os.path.basename(path)

        # this is for meditation
        subject = '_'.join(fname.split('_')[:2])

        print("Using MRI subject: " + subject)

        trans = os.path.join(folder, subject + '-trans.fif')

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

        if dr_method == 'pca':
            stc = create_vol_stc(
                raw=raw, 
                trans=trans, 
                subject=subject, 
                noise_cov=noise_cov, 
                spacing=vol_spacing,
                mne_method=mne_method,
                mne_depth=mne_depth,
                subjects_dir=subjects_dir) 

            data = stc.data

            # CROP TO MEDITATIONS
            data_array = []
            for key, ivals in intervals.items():
                if key == 'mind':
                    for ival in ivals:
                        start_idx = int((ival[0]) * sampling_rate_raw)
                        end_idx = int((ival[1]) * sampling_rate_raw)
                        data_array.append(data[:, start_idx:end_idx])

            data = np.concatenate(data_array, axis=-1)

            print("Original data shape: " + str(data.shape)) 
            print("Prepare using hilbert")
            data = prepare_hilbert(data, sampling_rate_raw,
                                   sampling_rate_hilbert)

            vertices = stc.vertices
        elif dr_method == 'ch_average':
            label_stc = create_vol_stc_labels(
                raw=raw, 
                trans=trans, 
                subject=subject, 
                noise_cov=noise_cov, 
                spacing=vol_spacing,
                mne_method=mne_method,
                mne_depth=mne_depth,
                subjects_dir=subjects_dir) 

            data = label_stc.data

            # CROP TO MEDITATIONS
            data_array = []
            for key, ivals in intervals.items():
                if key == 'mind':
                    for ival in ivals:
                        start_idx = int((ival[0]) * sampling_rate_raw)
                        end_idx = int((ival[1]) * sampling_rate_raw)
                        data_array.append(data[:, start_idx:end_idx])

            data = np.concatenate(data_array, axis=-1)

            print("Original data shape: " + str(data.shape)) 
            print("Prepare using hilbert")
            data = prepare_hilbert(data, sampling_rate_raw,
                                   sampling_rate_hilbert)
             
            vox_read = 0
            rows = []
            for idx in range(len(label_stc.vertices)):
                start = vox_read
                end = len(label_stc.vertices[idx])
                vox_read += len(label_stc.vertices[idx])
                rows.append(np.mean(data[start:end], axis=0))
            data = np.array(rows)
                
        # zscore
        # data = (data - np.mean(data)) / np.std(data)

        subject_item['data'] = data

        current_time += data.shape[-1] / sampling_rate_hilbert

        subject_data.append(subject_item)

    if dr_method == 'pca':
        brain_data = []
        means = []
        stds = []
        for subject in subject_data:
            data = subject['data']
            mean = np.mean(data, axis=-1)
            std = np.std(data, axis=-1)
            means.append(mean)
            stds.append(std)
            data = (data - mean[..., np.newaxis]) / std[..., np.newaxis]
            brain_data.append(data)
        brain_data = np.concatenate(brain_data, axis=-1)
        gstd = np.sqrt(np.mean(np.power(stds, 2), axis=0))
        gmean = np.mean(means, axis=0)
        brain_data = (brain_data * gstd[..., np.newaxis]) + gmean[..., np.newaxis]

    elif dr_method == 'ch_average':
        brain_data = []
        for subject in subject_data:
            data = subject['data']
            # data = ((data - np.mean(data, axis=-1)[:, np.newaxis]) / 
            #         np.std(data, axis=-1)[:, np.newaxis])
            data = ((data - np.mean(data)) / 
                    np.std(data))

            brain_data.append(data)
        brain_data = np.concatenate(brain_data, axis=-1)

    if dr_method == 'pca':
        brain_pca = sklearn.decomposition.PCA(n_components=20, whiten=True)
        brain_data_dr = brain_pca.fit_transform(brain_data.T).T
    elif dr_method == 'ch_average':
        brain_data_dr = brain_data

    # plot the dimension reducted data
    fig, ax = plt.subplots(nrows=brain_data_dr.shape[0], ncols=1)
    for idx in range(brain_data_dr.shape[0]):
        ax[idx].plot(brain_data_dr[idx])

    if save_path:
        series_path = os.path.join(save_path, 'series')
        if not os.path.exists(series_path):
            os.makedirs(series_path)
        fig.savefig(os.path.join(series_path, 'data_dr.png'), dpi=620)

    from hmmlearn.hmm import GaussianHMM
    markov = GaussianHMM(
        n_components=6,
        covariance_type='full')

    markov.fit(brain_data_dr.T)

    if dr_method == 'pca':

        # plot pure maps
        for comp_idx in range(brain_pca.components_.shape[0]):
            component = brain_pca.components_[comp_idx]
            plot_vol_stc_brainmap(save_path, 'pure_' + str(comp_idx+1).zfill(2), component, 
                                  vertices, vol_spacing, subjects_dir)

        # plot state maps
        for state_idx in range(markov.means_.shape[0]):
            weights = markov.means_[state_idx]
            print("For state " + str(state_idx+1) + " the weights are: ")
            print(str(weights))
            statemap = np.dot(weights, brain_pca.components_)
            plot_vol_stc_brainmap(save_path, 'state_' + str(state_idx+1).zfill(2), statemap, 
                                  vertices, vol_spacing, subjects_dir)

    # plot time series
    state_chain = markov.predict(brain_data_dr.T)

    nrows = len(subject_data)
    colors = plt.cm.get_cmap('gist_rainbow', markov.means_.shape[0])

    fig, axes = plt.subplots(ncols=1, nrows=nrows)
    for row_idx in range(nrows):
        start = int(subjects[row_idx]['start'] * float(sampling_rate_hilbert))
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

