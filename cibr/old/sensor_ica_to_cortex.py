import sys
import argparse

import pyface.qt

import mne
import numpy as np
import matplotlib.pyplot as plt 

import scipy.signal

from signals.ica.complex_ica import complex_ica

from signals.common import load_raw
from signals.common import preprocess
from signals.common import cortical_projection
from signals.common import calculate_stft
from signals.common import calculate_istft
from signals.common import arrange_as_matrix
from signals.common import arrange_as_tensor
from signals.common import compute_raw_intervals
from signals.common import combine_raw_intervals


def visualize_components_ts(data, mixing, dewhitening, sfreq, mean, raw_info):
    ch_names = ['ICA ' + str(i) for i in range(data.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='grad')
    raw = mne.io.RawArray(data, info)
    raw.plot(scalings='auto')

    brainmaps = np.abs(np.dot(dewhitening, mixing) + mean[:, np.newaxis])

    fig_ = plt.figure()
    for i in range(brainmaps.shape[1]):

        topo_data = brainmaps[:, i]
        axes = fig_.add_subplot(10, (brainmaps.shape[1] - 1) / 10 + 1, i+1)

        mne.viz.plot_topomap(topo_data, raw_info, axes=axes, show=False)

    psds, freqs = mne.time_frequency.psd_welch(raw, fmin=5, fmax=14, n_fft=1024)

    fig_ = plt.figure()
    for i in range(psds.shape[0]):

        axes = fig_.add_subplot(10, (psds.shape[0] - 1) / 10 + 1, i+1)
        axes.plot(freqs, psds[i])

    plt.show(block=False)


def visualize_components_in_cortex(data, fwd, cov, mixing, dewhitening, 
                                   mean, sfreq, window, noverlap,
                                   istft_freq_pads, raw_info, stft_shape):

    while True:
        input_ = raw_input("Choose component to plot: ")
        try:
            input_as_int = int(input_) - 1
        except:
            break
        comp_data = data.copy()
        comp_data[:input_as_int, :] = 0
        comp_data[input_as_int+1:, :] = 0
        
        comp_data = (np.dot(dewhitening, np.dot(mixing, comp_data)) + 
                     mean[:, np.newaxis])

        comp_data = arrange_as_tensor(comp_data, stft_shape)

        comp_data = calculate_istft(comp_data, sfreq, window, noverlap,
                                    istft_freq_pads)
        comp_raw = mne.io.RawArray(comp_data, raw_info)

        stc = cortical_projection(comp_raw, fwd, cov)

        fmin = np.percentile(stc.data, 95)
        fmid = np.percentile(stc.data, 97)
        fmax = np.percentile(stc.data, 98)

        brain = stc.plot(initial_time=300.0, hemi='both', clim={'kind': 'value', 'lims': [fmin, fmid, fmax]})
        brain.show_view(view='parietal')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raws', nargs='+')
    parser.add_argument('--fwd')
    parser.add_argument('--empty')
    cli_args = parser.parse_args()

    raw = load_raw(cli_args.raws)

    sfreq = raw.info['sfreq']
    window_in_seconds = 2
    n_components = 20
    conveps = 1e-7
    maxiter = 2000
    hpass = 4
    lpass = 17
    window_in_samples = np.power(2, np.ceil(np.log(
        sfreq * window_in_seconds)/np.log(2)))
    overlap_in_samples = (window_in_samples * 3) / 4

    raw, events = preprocess(raw)

    intervals = compute_raw_intervals(raw, events)

    # select only meditation
    raw = combine_raw_intervals(raw, intervals[10])

    freqs, times, data, istft_freq_pads = (
        calculate_stft(raw._data, sfreq, window_in_samples, 
                       overlap_in_samples, hpass, lpass))

    shape, data = data.shape, arrange_as_matrix(data)

    data, mixing, dewhitening, _, _, mean = complex_ica(
        data, n_components, conveps=conveps, maxiter=maxiter)

    ica_ts = calculate_istft(arrange_as_tensor(data, shape), sfreq, 
        window_in_samples, overlap_in_samples, istft_freq_pads)

    visualize_components_ts(ica_ts, mixing, dewhitening, sfreq, mean, raw.info)

    fwd = mne.forward.read_forward_solution(cli_args.fwd)
    cov = mne.compute_raw_covariance(mne.io.Raw(cli_args.empty, preload=True).filter(l_freq=hpass, h_freq=lpass))

    visualize_components_in_cortex(data, fwd, cov, mixing, dewhitening, mean,
                                   sfreq, window_in_samples, overlap_in_samples,
                                   istft_freq_pads, raw.info, shape)

    import pdb; pdb.set_trace()
    print "miau"


