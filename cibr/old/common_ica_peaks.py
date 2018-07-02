import sys
import os

import argparse

import mne
import numpy as np

import matplotlib.pyplot as plt

from signals.ica.complex_ica import complex_ica

from signals.common import calculate_stft
from signals.common import arrange_as_matrix
from signals.common import arrange_as_tensor
from signals.common import load_raw
from signals.common import preprocess
from signals.common import combine_stft_intervals
from signals.common import compute_stft_intervals
from signals.common import get_peak
from signals.common import get_power

import matplotlib

# offline rendering
matplotlib.use('Agg')
 
matplotlib.rc('font', size=6)



def plot_components(freqs, raw_times, splits_in_samples, data, mixing, dewhitening, mean, save_path, page, raw_info, normalize=False):

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # plot brainmaps
    brainmaps  = np.abs(np.dot(dewhitening, mixing) + mean[:, np.newaxis])

    fig_ = plt.figure()
    for i in range(brainmaps.shape[1]):

        topo_data = brainmaps[:, i]
        axes = fig_.add_subplot(page, (brainmaps.shape[1] - 1) / page + 1, i+1)
        mne.viz.plot_topomap(topo_data, raw_info, axes=axes, show=False)

    fig_.savefig(save_path + 'topo.png', dpi=310)

    # plot spectra
    fig_ = plt.figure()
    for i in range(data.shape[0]):

        y = np.mean(np.abs(data[i]), axis=-1)
        x = freqs
        axes = fig_.add_subplot(page, (data.shape[0] - 1) / page + 1, i+1)
        axes.plot(x,y)

    fig_.savefig(save_path + 'spectra.png', dpi=310)

    # plot spectra for different files

    psds = []

    for idx in range(len(splits_in_samples) - 1):

        smin = int((splits_in_samples[idx] / float(len(raw_times))) * 
                   data.shape[2] + 0.5)
        smax = int((splits_in_samples[idx+1] / float(len(raw_times))) * 
                   data.shape[2] + 0.5)
        
        tfrs = (np.abs(data))[:, :, smin:smax]
        psds.append(np.mean(tfrs, axis=-1))

    psds = np.array(psds)
    if normalize:
        for i in range(psds.shape[0]):
            for j in range(psds.shape[1]):
                psds[i, j] = (psds[i, j] - np.min(psds[i,j])) / np.max(psds[i,j])

    fig_ = plt.figure()
    for i in range(data.shape[0]):

        ax = fig_.add_subplot(page, (data.shape[0] - 1) / page + 1, i+1)
        for psd in psds:
            ax.plot(freqs, psd[i])
        ax.set_xlabel = 'Frequency (Hz)'
        ax.set_ylabel = 'Power (dB)'

    fig_.savefig(save_path + 'spectra_subjects.png', dpi=310)

    # plt.show()


def save_values(raw, freqs, event_id, event_name, data, save_path, power=False):

    intervals = compute_stft_intervals(data, events, raw.first_samp,
        len(raw.times))[event_id]
    combined_data = combine_stft_intervals(data, intervals)

    psds = np.abs([np.mean(stft, axis=-1) for stft in combined_data])

    if power:
        values  = np.array([get_power(freqs, psd) for psd in psds])
    else:
        values  = np.array([get_peak(freqs, psd)[1] for psd in psds])

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    savename = save_path + event_name + '_peak_values.csv'
    np.savetxt(savename, values)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('save_path')
    parser.add_argument('--raws', nargs='+')
    cli_args = parser.parse_args()

    print "Processing files: "

    raws = []
    splits_in_samples = [0]
    for path_idx, path in enumerate(cli_args.raws):

        print path

        raw = load_raw(path)
        raw, events = preprocess(raw)

        # normalize
        raw._data = (raw._data - np.mean(raw._data)) / np.std(raw._data)

        for idx, event in enumerate(events):
            if event[2] == 11:
                raws.append(raw.copy().crop(
                    tmin=((event[0] - raw.first_samp)/raw.info['sfreq']), 
                    tmax=((events[idx+1][0] - raw.first_samp)/raw.info['sfreq'])))

                splits_in_samples.append(splits_in_samples[-1] + events[idx+1][0] - events[idx][0])
                break

    for split in splits_in_samples:
        print split

    raw = mne.concatenate_raws(raws)

    sfreq = raw.info['sfreq']
    page = 10
    window_in_seconds = 4
    n_components = 40
    conveps = 1e-8
    maxiter = 12000
    hpass = 4
    lpass = 24
    window_in_samples = np.power(2, np.ceil(np.log(
        sfreq * window_in_seconds)/np.log(2)))
    overlap_in_samples = (window_in_samples * 3) / 4

    freqs, times, data, _ = calculate_stft(raw._data, sfreq, window_in_samples,
                                           overlap_in_samples, hpass, lpass)

    shape, data = data.shape, arrange_as_matrix(data)

    data, mixing, dewhitening, unmixing, whitening, mean = complex_ica(
        data, n_components, conveps=conveps, maxiter=maxiter)

    data = arrange_as_tensor(data, shape)

    common_save_path = cli_args.save_path + '/common/'
    plot_components(freqs, raw.times, splits_in_samples, data, mixing, dewhitening, 
                    mean, common_save_path, page, raw.info, normalize=True)

    # extraction part
    
    for path_idx, path in enumerate(cli_args.raws):
        raw = load_raw(path)
        raw, events = preprocess(raw)

        # take the number out of `/path/to/subject_001_block1.fif` type of name
        subject_name = path.split('/')[-1].split('_')[1]


        # normalize
        raw._data = (raw._data - np.mean(raw._data)) / np.std(raw._data)

        freqs, times, data, _ = calculate_stft(raw._data, sfreq, window_in_samples,
                                               overlap_in_samples, hpass, lpass)

        shape, data = data.shape, arrange_as_matrix(data)

        # extract component stft's

        data = np.dot(np.dot(unmixing, whitening), (data - mean[:, np.newaxis]))

        data = arrange_as_tensor(data, shape)

        save_path = cli_args.save_path + 'peaks/' + subject_name + '/'

        save_values(raw, freqs, 10, 'mind', data, save_path, power=False)
        save_values(raw, freqs, 11, 'rest', data, save_path, power=False)
        save_values(raw, freqs, 12, 'plan', data, save_path, power=False)
        save_values(raw, freqs, 13, 'anx', data, save_path, power=False)

        save_path = cli_args.save_path + 'power/' + subject_name + '/'

        save_values(raw, freqs, 10, 'mind', data, save_path, power=True)
        save_values(raw, freqs, 11, 'rest', data, save_path, power=True)
        save_values(raw, freqs, 12, 'plan', data, save_path, power=True)
        save_values(raw, freqs, 13, 'anx', data, save_path, power=True)
