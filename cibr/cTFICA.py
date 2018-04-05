import sys
import os
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
from signals.common import arrange_as_matrix
from signals.common import arrange_as_tensor
from signals.common import combine_raw_intervals
from signals.common import compute_raw_intervals


def compute_src_intervals():
    pass


def combine_src_intervals():
    pass


def visualize_components(data, mixing, dewhitening, mean, freqs, vertices):
    brainmaps = np.abs(np.dot(dewhitening, mixing) + mean[:, np.newaxis])
    spectrums = np.mean(np.power(np.abs(data), 2), axis=2)

    # plot all spectrums
    fig_ = plt.figure()
    page = 10
    for i in range(spectrums.shape[0]):
        y = spectrums[i]
        x = freqs
        axes = fig_.add_subplot(page, (spectrums.shape[0] - 1) / page + 1, i+1)
        axes.plot(x,y)

        plt.show(block=False)

    while True:
        input_ = raw_input("Choose component to plot: ")
        try:
            input_as_int = int(input_) - 1
        except:
            break

        subject = os.environ['SUBJECT']

        brainmap_stc = mne.SourceEstimate(brainmaps[:, input_as_int, np.newaxis], 
            vertices=vertices, tmin=0, tstep=1, subject=subject)
        brain = brainmap_stc.plot(hemi='both')
        brain.show_view(view='parietal')
        

    import pdb; pdb.set_trace()
    print "miau"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raws', nargs='+')
    parser.add_argument('--fwd')
    parser.add_argument('--empty')
    cli_args = parser.parse_args()

    raw = load_raw(cli_args.raws)

    sfreq = raw.info['sfreq']
    window_in_seconds = 2
    n_components = 30
    conveps = 1e-10
    maxiter = 15000
    hpass = 4
    lpass = 30
    window_in_samples = np.power(2, np.ceil(np.log(
        sfreq * window_in_seconds)/np.log(2)))
    overlap_in_samples = (window_in_samples * 3) / 4

    raw, events = preprocess(raw)

    intervals = compute_raw_intervals(raw, events)
    meditation_raw = combine_raw_intervals(raw, intervals[10][:4])
 
    fwd = mne.forward.read_forward_solution(cli_args.fwd)
    cov = mne.compute_raw_covariance(mne.io.Raw(cli_args.empty, preload=True))
    # cov = mne.compute_raw_covariance(combine_raw_intervals(raw, intervals[11]))

    stc = cortical_projection(meditation_raw, fwd, cov)

    freqs, times, data, _ = calculate_stft(stc.data, sfreq, window_in_samples, 
                                        overlap_in_samples, hpass, lpass)

    shape, data = data.shape, arrange_as_matrix(data)

    data, mixing, dewhitening, _, _, mean = complex_ica(
        data, n_components, conveps=conveps, maxiter=maxiter)

    data = arrange_as_tensor(data, shape)

    visualize_components(data, mixing, dewhitening, mean, freqs, stc.vertices)

    import pdb; pdb.set_trace()
    print "miau"


