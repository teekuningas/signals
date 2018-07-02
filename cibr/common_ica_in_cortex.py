PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('font', size=6)
    matplotlib.use('Agg')

import sys
import argparse
import os

import pyface.qt

import mne
import numpy as np
import matplotlib.pyplot as plt 

import scipy.signal

from signals.cibr.ica.complex_ica import complex_ica

from signals.cibr.common import load_raw
from signals.cibr.common import preprocess
from signals.cibr.common import get_correlations
from signals.cibr.common import calculate_stft
from signals.cibr.common import arrange_as_matrix
from signals.cibr.common import arrange_as_tensor
from signals.cibr.common import plot_mean_spectra


def create_stc(raw, trans, subject, empty_raw, subjects_dir,
               window_in_seconds, hpass, lpass):
    """
    """
    mri = os.path.join(subjects_dir, 'fsaverage', 'mri',
                       'T1.mgz')

    bem = os.path.join(subjects_dir, subject, 'bem',
                       subject+'-inner_skull-bem-sol.fif')

    # create volumetric source space
    # src = mne.setup_volume_source_space(
    #     subject=subject,
    #     bem=bem,
    #     mri=mri,
    #     subjects_dir=subjects_dir)

    # create surface-based source space
    src = mne.setup_source_space(
        subject=subject,
        spacing='ico4',
        surface='white',
        subjects_dir=subjects_dir)

    fwd = mne.make_forward_solution(
        info=raw.info, 
        trans=trans, 
        src=src,
        bem=bem,
        meg=True,
        eeg=False)

    noise_cov = mne.compute_raw_covariance(empty_raw, tmin=4)

    inv = mne.minimum_norm.make_inverse_operator(
        info=raw.info,
        forward=fwd,
        noise_cov=noise_cov,
        loose=0.2,
        depth=0.8,
        fixed=False,
        limit_depth_chs=False)

    sfreq = raw.info['sfreq']
    window_in_samples = np.power(2, np.ceil(np.log(
        sfreq * window_in_seconds)/np.log(2)))
    overlap_in_samples = (window_in_samples * 3) / 4

    freqs, times, data, _ = calculate_stft(
        raw._data, 
        sfreq, 
        window_in_samples, 
        overlap_in_samples, 
        hpass, 
        lpass, 
        row_wise=True)

    shape, data = data.shape, arrange_as_matrix(data)

    raw = mne.io.RawArray(data, raw.info)

    stc = mne.minimum_norm.apply_inverse_raw(
        raw=raw,
        inverse_operator=inv,
        lambda2=0.1,
        method='MNE',
        label=None,
        pick_ori='normal')

    # import pdb; pdb.set_trace()

    return stc, freqs, times, shape


def plot_stc_brainmaps():
    pass


def extract_intervals(events):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raws', nargs='+')
    parser.add_argument('--empty')
    parser.add_argument('--save-path')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    window_in_seconds = 2
    n_components = 30
    page = 10
    conveps = 1e-6
    maxiter = 15000
    hpass = 4
    lpass = 17

    # process similarly to input data 
    empty_raw = load_raw(cli_args.empty)
    empty_raw.resample(50)
    empty_raw, _ = preprocess(empty_raw, filter_=(2, 20))

    empty_raw._data = (
        (empty_raw._data - np.mean(empty_raw._data)) / 
        np.std(empty_raw._data))

    subject_data = []
    for path_idx, path in enumerate(cli_args.raws):
        folder = os.path.dirname(path)
        fname = os.path.basename(path)
        subject = fname.split('.fif')[0]
        trans = os.path.join(folder, subject + '-trans.fif')

        print "Handling ", path

        raw = load_raw(path)
        raw.resample(50)
        raw, events = preprocess(raw, filter_=(2, 20))

        intervals = extract_intervals(events)

        raw._data = (raw._data - np.mean(raw._data)) / np.std(raw._data)

        stc, freqs, times, shape = create_stc(
            raw=raw, 
            trans=trans, 
            subject=subject, 
            empty_raw=empty_raw, 
            subjects_dir=subjects_dir, 
            window_in_seconds=window_in_seconds, 
            hpass=hpass, 
            lpass=lpass)

        subject_item = {}
        subject_item['name'] = subject
        subject_item['stc'] = stc
        subject_item['freqs'] = freqs
        subject_item['times'] = times
        subject_item['shape'] = shape
        subject_item['intervals'] = intervals

        subject_data.append(subject_item)

    concat_stc_data = np.concatenate(
        [subject['stc'].data for subject in subject_data], axis=1)

    for subject in subject_data:
        del subject['stc']

    concat_shape = (shape[0], shape[1], 
                    sum([subject['shape'][2] for subject in subject_data]))

    data, mixing, dewhitening, _, _, mean = complex_ica(
        concat_stc_data, n_components, conveps=conveps, maxiter=maxiter)

    data = arrange_as_tensor(data, concat_shape)

    print "Plotting mean spectra."
    plot_mean_spectra(save_path, data, freqs, page, range(n_components))

    import pdb; pdb.set_trace()

    print "Plotting brain maps."
    plot_stc_brainmaps(save_path, mixing, dewhitening, mean, stc.copy())

    # I need to have a way to get all the subjects separate and to get their heartbeat
    # and note intervals
    # Also I want to compare components between subjects that do well in the heartbeat
    # task to those that do bad. what is the difference between their brain responses
    # during the heartbeat task.
    

    import pdb; pdb.set_trace()
    print "miau"


