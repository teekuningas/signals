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
from signals.cibr.common import plot_brainmaps
from signals.cibr.common import plot_mean_spectra
from signals.cibr.common import plot_subject_spectra
from signals.cibr.common import get_rest_intervals


def visualize_components_in_cortex(inv, maps, raw_info, save_path):

    loop_idx = 0
    while True:
        if save_path:
            component_idx = loop_idx
        else:

            input_ = raw_input("Choose component to plot: ")
            try:
                component_idx = int(input_) - 1
            except:
                break

        try:
            evoked = mne.EvokedArray(maps[:, component_idx, np.newaxis],
                raw_info)
        except:
            break

        # stc = mne.minimum_norm.apply_inverse(evoked, inv)
        stc = mne.beamformer.apply_lcmv(evoked, inv)

        fmin = np.percentile(stc.data, 70)
        fmid = np.percentile(stc.data, 90)
        fmax = np.percentile(stc.data, 99)

        brain = stc.plot(hemi='split', views=['med', 'lat'], smoothing_steps=50,
                         surface='white',
                         clim={'kind': 'value', 'lims': [fmin, fmid, fmax]})

        # fig_lh = stc.plot(hemi='lh', views=('med', 'lat'), backend='matplotlib')
        # fig_rh = stc.plot(hemi='rh', views=('med', 'lat'), backend='matplotlib')

        if save_path:
            # lh_path = os.path.join(save_path, 'brains', 
            #     'comp_' + str(component_idx).zfill(2) + '_lh.png')
            # rh_path = os.path.join(save_path, 'brains', 
            #     'comp_' + str(component_idx).zfill(2) + '_rh.png')

            brain_path = os.path.join(save_path, 'brains')
            if not os.path.exists(brain_path):
                os.makedirs(brain_path)

            path = os.path.join(brain_path, 
                'comp_' + str(component_idx+1).zfill(2) + '.png')

            # fig_rh.savefig(rh_path, dpi=310)
            # fig_lh.savefig(lh_path, dpi=310)

            brain.save_image(path)

        loop_idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raws', nargs='+')
    parser.add_argument('--fwd')
    parser.add_argument('--data')
    parser.add_argument('--empty')
    parser.add_argument('--save-path')
    cli_args = parser.parse_args()

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    raws = []
    names = []
    splits_in_samples = [0]
    for path_idx, path in enumerate(cli_args.raws):

        print path

        raw = load_raw(path)
        raw, events = preprocess(raw, filter_=False)

        raw._data = (raw._data - np.mean(raw._data)) / np.std(raw._data)

        raws.append(raw)
        names.append(raw.filenames[0].split('/')[-1].split('.fif')[0])

        splits_in_samples.append(splits_in_samples[-1] + len(raw))

    raw = mne.concatenate_raws(raws)

    sfreq = raw.info['sfreq']
    window_in_seconds = 2
    n_components = 30
    page = 10
    conveps = 1e-7
    maxiter = 15000
    hpass = 4
    lpass = 16
    window_in_samples = np.power(2, np.ceil(np.log(
        sfreq * window_in_seconds)/np.log(2)))
    overlap_in_samples = (window_in_samples * 3) / 4

    eo_ivals, ec_ivals, total_ivals = get_rest_intervals(splits_in_samples, sfreq)

    # create new data based only on ec
    raws = []
    index_count = 0
    total_ivals = []
    for ival in ec_ivals:
        total_ival = (index_count, index_count + ival[1] + ival[0])
        index_count += ival[1] - ival[0]
        total_ivals.append(total_ival)
        raws.append(mne.io.RawArray(raw._data[:, ival[0]:ival[1]], raw.info))

    raw = mne.concatenate_raws(raws)
    print total_ivals

    freqs, times, data, _ = (
        calculate_stft(raw._data, sfreq, window_in_samples, 
                       overlap_in_samples, hpass, lpass))

    shape, data = data.shape, arrange_as_matrix(data)

    data, mixing, dewhitening, _, _, mean = complex_ica(
        data, n_components, conveps=conveps, maxiter=maxiter)

    data = arrange_as_tensor(data, shape)

    print "Calculating correlations."
    correlations = get_correlations(data, freqs, total_ivals, raw.times)
    corr_scores = np.sum(correlations, axis=1)

    # sort in reverse order
    corr_idxs = np.argsort(-corr_scores)

    print "Corr scores sorted: "
    for idx in corr_idxs:
        print 'Component ' + str(idx+1) + ': ' + str(corr_scores[idx])

    print "Plotting brainmaps."
    plot_brainmaps(save_path, dewhitening, mixing, mean, raw.info,
                   page, corr_idxs)
    # print "Plotting mean spectra."
    # plot_mean_spectra(save_path, data, freqs, page, corr_idxs)
    # print "Plotting subject spectra."
    # plot_subject_spectra(save_path, data, freqs, page,
    #                      total_ivals, raw.times, corr_idxs)

    fwd = mne.forward.read_forward_solution(cli_args.fwd)

    empty_raw = mne.io.Raw(cli_args.empty, preload=True)
    empty_raw = empty_raw.resample(raw.info['sfreq'])
    empty_raw.filter(l_freq=raw.info['highpass'], h_freq=raw.info['lowpass'])
    noise_cov = mne.compute_raw_covariance(empty_raw)

    data_raw = mne.io.Raw(cli_args.data, preload=True)
    data_raw = data_raw.resample(raw.info['sfreq'])
    data_raw.filter(l_freq=raw.info['highpass'], h_freq=raw.info['lowpass'])
    data_cov = mne.compute_raw_covariance(data_raw)

    # data_cov = mne.compute_raw_covariance(raw)

    # inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov)
    inv = mne.beamformer.make_lcmv(raw.info, fwd, data_cov, reg=0.20, noise_cov=noise_cov, pick_ori=None)

    maps = np.abs(np.matmul(dewhitening, mixing) + mean[:, np.newaxis])[:, corr_idxs]

    print "Plotting brains"
    visualize_components_in_cortex(inv, maps, raw.info, save_path)

    import pdb; pdb.set_trace()
    print "miau"


