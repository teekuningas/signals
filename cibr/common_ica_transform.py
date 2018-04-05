import matplotlib
matplotlib.rc('font', size=6)
matplotlib.use('Agg')

import sys
import os
import csv

import argparse

import mne
import numpy as np
import scipy

import matplotlib.pyplot as plt

from signals.ica.complex_ica import complex_ica

from signals.common import preprocess
from signals.common import load_raw
from signals.common import calculate_stft
from signals.common import arrange_as_matrix
from signals.common import arrange_as_tensor
from signals.common import get_power_law


def get_mean_spectra(data, freqs):
    mean_spectra = []  
    for i in range(data.shape[0]):
        spectrum = np.mean(np.abs(data[i]), axis=-1)
        plaw = get_power_law(freqs, spectrum)
        psd = spectrum - plaw
        psd = (psd - np.min(psd)) / (np.max(psd) - np.min(psd))
        mean_spectra.append(psd)

    return np.array(mean_spectra)


def get_rest_intervals(splits_in_samples, sfreq):
    eyes_open = []
    eyes_closed = []
    total = []

    for idx in range(len(splits_in_samples) - 1):
        subject_start = splits_in_samples[idx]
        subject_end = splits_in_samples[idx+1]
        eo_ival = (subject_start + 15*sfreq, 
                   (subject_end + subject_start) / 2 - 15*sfreq)
        ec_ival = ((subject_end + subject_start) / 2 + 15*sfreq,
                   subject_end - 15*sfreq)
        total_ival = (subject_start + 15*sfreq, 
                      subject_end - 15*sfreq)

        eyes_open.append(eo_ival)
        eyes_closed.append(ec_ival)
        total.append(total_ival)

    return eyes_open, eyes_closed, total

def get_subject_spectra(data, freqs, intervals, raw_times, normalized=True, subtract_power_law=True):

    subject_spectra = []
    for i in range(data.shape[0]):
        subject_spectra_i = []
        for ival in intervals:
            smin = int((ival[0] / float(len(raw_times))) * data.shape[2] + 0.5)
            smax = int((ival[1] / float(len(raw_times))) * data.shape[2] + 0.5)
            
            tfr = (np.abs(data))[i, :, smin:smax]
            subject_spectra_i.append(np.mean(tfr, axis=-1))
        subject_spectra.append(subject_spectra_i)

    subject_spectra = np.array(subject_spectra)

    for i in range(subject_spectra.shape[0]):
        for j in range(subject_spectra.shape[1]):

            psd_ij = subject_spectra[i, j]

            if subtract_power_law:
                plaw = get_power_law(freqs, subject_spectra[i, j])
                psd_ij = psd_ij - plaw

            if normalized:
                psd_ij = ((psd_ij - np.min(psd_ij)) / 
                          (np.max(psd_ij) - np.min(psd_ij)))

            subject_spectra[i, j] = psd_ij

    return subject_spectra


def roll_subject_to_average(subject_psd, average_psd):
    x, y = subject_psd, average_psd
    corr = np.correlate(x, y, mode='full')
    max_corr = np.argmax(corr)
    displacement = max_corr - len(x) + 1
    padded_x = np.pad(x, (max(0, displacement), -min(0, displacement)),
                      mode='edge')
    rolled_x = np.roll(padded_x, -displacement)
    moved_x = rolled_x[max(0, displacement):len(rolled_x)+min(0, displacement)]
    return moved_x


def get_peak_by_correlation(subject_psd, average_psd):

    avg_argmax = np.argmax(average_psd)
    corr = np.correlate(subject_psd, average_psd, mode='full')
    displacement = np.argmax(corr) - len(subject_psd) + 1

    try:
        test_val = subject_psd[avg_argmax + displacement]
        return avg_argmax + displacement
    except:
        return avg_argmax


def get_correlations(data, freqs, intervals, raw_times):

    mean_spectra = get_mean_spectra(data, freqs)
    subject_spectra = get_subject_spectra(data, freqs, intervals, raw_times)

    correlations = []
    for i in range(data.shape[0]):
        correlations_i = []
        for j in range(len(splits_in_samples) - 1):
            ts1 = mean_spectra[i]
            ts2 = subject_spectra[i, j]

            ts2 = roll_subject_to_average(ts2, ts1)
 
            r, _ = scipy.stats.pearsonr(ts1, ts2) 
            correlations_i.append(r)
        correlations.append(correlations_i)
    correlations = np.array(correlations)

    return correlations


def plot_brainmaps(save_path, dewhitening, mixing, mean, raw_info, page, component_idxs):

    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # extract brainmaps from the mixing matrix
    brainmaps  = np.abs(np.dot(dewhitening, mixing) + mean[:, np.newaxis])

    # plot brainmaps of selected (ordered) component indices
    fig_ = plt.figure()
    for i, idx in enumerate(component_idxs):
        topo_data = brainmaps[:, idx]
        axes = fig_.add_subplot(page, (len(component_idxs) - 1) / page + 1, i+1)
        mne.viz.plot_topomap(topo_data, raw_info, axes=axes, show=False)

    # save plotted maps
    if save_path:
        fig_.savefig(os.path.join(save_path, 'topo.png'), dpi=310)


def plot_mean_spectra(save_path, data, freqs, page, component_idxs):

    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # extract mean spectra
    mean_spectra = get_mean_spectra(data, freqs)

    # plot mean spectra of selected (ordered) component indices
    fig_ = plt.figure()
    for i, idx in enumerate(component_idxs):

        spectrum = mean_spectra[idx]
        axes = fig_.add_subplot(page, (len(component_idxs) - 1) / page + 1, i+1)
        axes.plot(freqs, spectrum)
        axes.set_xlabel = 'Frequency (Hz)'
        axes.set_ylabel = 'Power (dB)'

    # save plotted spectra
    if save_path:
        fig_.savefig(os.path.join(save_path, 'spectra.png'), dpi=310)


def plot_subject_spectra(save_path, data, freqs, page, intervals, raw_times, component_idxs):

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # extract mean spectra
    mean_spectra = get_mean_spectra(data, freqs)

    # extract spectra for different subjects
    subject_spectra = get_subject_spectra(data, freqs, intervals, raw_times)

    # plot subject spectra of selected (ordered) component indices
    fig_ = plt.figure()
    for i, idx in enumerate(component_idxs):
        ax = fig_.add_subplot(page, (len(component_idxs) - 1) / page + 1, i+1)

        for psd in subject_spectra[idx]:
            spectrum = roll_subject_to_average(psd, mean_spectra[idx])
            ax.plot(freqs, spectrum)
            ax.set_xlabel = 'Frequency (Hz)'
            ax.set_ylabel = 'Power (dB)'

    # save plotted subject spectra
    if save_path:
        fig_.savefig(os.path.join(save_path, 'spectra_subjects.png'), dpi=310)


def plot_subject_spectra_separate(save_path, data, freqs, page, intervals, raw_times, component_idxs, names):

    if save_path:
        save_path = os.path.join(save_path, 'separate')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # extract spectra for different subjects
    subject_spectra = get_subject_spectra(data, freqs, intervals, raw_times)

    # plot subject spectra of selected (ordered) component indices
    for sub_idx in range(subject_spectra.shape[1]):
        fig_ = plt.figure()
        for i, comp_idx in enumerate(component_idxs):
            ax = fig_.add_subplot(page, (len(component_idxs) - 1) / page + 1, i+1)

            spectrum = subject_spectra[comp_idx, sub_idx]
            ax.plot(freqs, spectrum)
            ax.set_xlabel = 'Frequency (Hz)'
            ax.set_ylabel = 'Power (dB)'

        # save plotted subject spectra
        if save_path:
            sub_path = os.path.join(save_path, names[sub_idx] + '.png')
            fig_.savefig(sub_path, dpi=310)


def save_peak_values(save_path, data, freqs, page, 
                     eo_ivals, ec_ivals, total_ivals, 
                     raw_times, component_idxs, correlations, names):

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # extract mean spectra
    mean_spectra = get_mean_spectra(data, freqs)

    # extract spectra for different subjects
    subject_spectra_normalized = get_subject_spectra(data, freqs, 
        total_ivals, raw_times, normalized=True)

    subject_spectra = get_subject_spectra(data, freqs, total_ivals, 
        raw_times, normalized=False)

    total_subject_spectra_ws = get_subject_spectra(data, freqs, 
        total_ivals, raw_times, normalized=False, 
        subtract_power_law=False)

    eo_subject_spectra_ws = get_subject_spectra(data, freqs, 
        eo_ivals, raw_times, normalized=False, 
        subtract_power_law=False)

    ec_subject_spectra_ws = get_subject_spectra(data, freqs, 
        ec_ivals, raw_times, normalized=False, 
        subtract_power_law=False)

    # prepare header
    header = ['Subject']
    for i in range(len(component_idxs)):
        header.append('Frequency (' + str(i+1) + ')')
        header.append('Total amplitude without powerlaw (' + str(i+1) + ')')
        header.append('Total amplitude (' + str(i+1) + ')')
        header.append('EO amplitude (' + str(i+1) + ')')
        header.append('EC amplitude (' + str(i+1) + ')')
        header.append('Score (' + str(i+1) + ')')
        
    data_array = []
    for sub_idx in range(subject_spectra.shape[1]):
        row = [names[sub_idx]]
        for comp_idx in component_idxs:
            psd_normalized = subject_spectra_normalized[comp_idx, sub_idx]
            peak_idx = get_peak_by_correlation(psd_normalized, mean_spectra[comp_idx])

            freq = freqs[peak_idx]

            total_peak = subject_spectra[comp_idx, sub_idx][peak_idx]
            total_peak_ws = total_subject_spectra_ws[comp_idx, sub_idx][peak_idx]  # noqa
            eo_peak_ws = eo_subject_spectra_ws[comp_idx, sub_idx][peak_idx]
            ec_peak_ws = ec_subject_spectra_ws[comp_idx, sub_idx][peak_idx]

            row.append(freq)
            row.append(total_peak)
            row.append(total_peak_ws)
            row.append(eo_peak_ws)
            row.append(ec_peak_ws)
            row.append(correlations[comp_idx, sub_idx])

        data_array.append(row)

    with open(os.path.join(save_path, 'data.csv'),'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        for line in data_array:
            writer.writerow(line)

def crop_function(raw, events):
    return raw

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('save_path')
    parser.add_argument('--raws', nargs='+')
    cli_args = parser.parse_args()

    print "Processing files: "

    raws = []
    names = []
    splits_in_samples = [0]
    for path_idx, path in enumerate(cli_args.raws):

        print path

        raw = load_raw(path)
        raw, events = preprocess(raw, filter_=False)

        raw._data = (raw._data - np.mean(raw._data)) / np.std(raw._data)

        raw = crop_function(raw, events)
        raws.append(raw)
        names.append(raw.filenames[0].split('/')[-1].split('.fif')[0])

        splits_in_samples.append(splits_in_samples[-1] + len(raw))

    raw = mne.concatenate_raws(raws)

    sfreq = raw.info['sfreq']
    page = 10
    window_in_seconds = 2
    n_components = 30
    conveps = 1e-7
    maxiter = 15000
    hpass = 4
    lpass = 16
    window_in_samples = np.power(2, np.ceil(np.log(
        sfreq * window_in_seconds)/np.log(2)))
    overlap_in_samples = (window_in_samples * 3) / 4

    eo_ivals, ec_ivals, total_ivals = get_rest_intervals(splits_in_samples, sfreq)

    print "EO ivals:"
    print eo_ivals
    print "EC ivals:"
    print ec_ivals
    print "Total ivals:"
    print total_ivals

    freqs, times, data, _ = calculate_stft(raw._data, sfreq, window_in_samples,
        overlap_in_samples, hpass, lpass, row_wise=True)

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
    plot_brainmaps(cli_args.save_path, dewhitening, mixing, mean, raw.info,
                   page, corr_idxs)
    print "Plotting mean spectra."
    plot_mean_spectra(cli_args.save_path, data, freqs, page, corr_idxs)
    print "Plotting subject spectra."
    plot_subject_spectra(cli_args.save_path, data, freqs, page, 
                         total_ivals, raw.times, corr_idxs)
    print "Plotting subject spectra to separate images"
    plot_subject_spectra_separate(cli_args.save_path, data, freqs, page, 
                                  total_ivals, raw.times, corr_idxs,
                                  names)

    print "Saving to data values."
    # save all the values in same order as the plot
    save_peak_values(cli_args.save_path, data, freqs, page, 
                     eo_ivals, ec_ivals, total_ivals,
                     raw.times, corr_idxs, correlations, names)

    import pdb; pdb.set_trace()
    print "miau"

