import sys
import os

import argparse

import mne
import numpy as np


from signals.ica.complex_ica import complex_ica

from signals.common import calculate_stft
from signals.common import arrange_as_matrix
from signals.common import arrange_as_tensor
from signals.common import load_raw
from signals.common import preprocess
from signals.common import combine_stft_intervals
from signals.common import compute_stft_intervals
from signals.common import get_peak

import matplotlib

# offline rendering
matplotlib.use('Agg')

# set font size
matplotlib.rc('font', size=6)

# must be after setting offline rendering
import matplotlib.pyplot as plt

def plot_components(freqs, data, mixing, dewhitening, mean, save_path, page, raw_info):

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


def save_peak_values(raw, freqs, event_id, event_name, data, save_path):

    intervals = compute_stft_intervals(data, events, raw.first_samp, 
        len(raw.times))[event_id]
    combined_data = combine_stft_intervals(data, intervals)
    
    psds = np.abs([np.mean(stft, axis=-1) for stft in combined_data])

    peaks = np.array([get_peak(freqs, psd)[1] for psd in psds])

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    savename = save_path + event_name + '_peak_values.csv'
    np.savetxt(savename, peaks)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('save_path')
    parser.add_argument('--raws', nargs='+')
    cli_args = parser.parse_args()

    raw = load_raw(cli_args.raws)

    raw, events = preprocess(raw)

    # normalize
    raw._data = (raw._data - np.mean(raw._data)) / np.std(raw._data)

    # take the number out of `/path/to/subject_001_block1.fif` type of name
    subject_name = cli_args.raws[0].split('/')[-1].split('_')[1]

    save_path = cli_args.save_path + subject_name + '/'

    sfreq = raw.info['sfreq']
    page = 5
    window_in_seconds = 2
    n_components = 20
    conveps = 1e-8
    maxiter = 2000
    hpass = 3
    lpass = 25
    window_in_samples = np.power(2, np.ceil(np.log(
        sfreq * window_in_seconds)/np.log(2)))
    overlap_in_samples = (window_in_samples * 3) / 4

    freqs, times, data, _ = calculate_stft(raw._data, sfreq, window_in_samples,
                                           overlap_in_samples, hpass, lpass)

    shape, data = data.shape, arrange_as_matrix(data)

    data, mixing, dewhitening, _, _, mean = complex_ica(
        data, n_components, conveps=conveps, maxiter=maxiter)

    data = arrange_as_tensor(data, shape)

    plot_components(freqs, data, mixing, dewhitening, mean, 
                    save_path, page, raw.info)

    save_peak_values(raw, freqs, 10, 'mind', data, save_path)
    save_peak_values(raw, freqs, 11, 'rest', data, save_path)
    save_peak_values(raw, freqs, 12, 'plan', data, save_path)
    save_peak_values(raw, freqs, 13, 'anx', data, save_path)

