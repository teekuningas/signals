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

from signals.cibr.ica.complex_ica import complex_ica
from sklearn.decomposition import FastICA

from signals.cibr.common import load_raw
from signals.cibr.common import preprocess
from signals.cibr.common import get_correlations
from signals.cibr.common import calculate_stft
from signals.cibr.common import arrange_as_matrix
from signals.cibr.common import arrange_as_tensor
from signals.cibr.common import plot_mean_spectra
from signals.cibr.common import plot_subject_spectra
from signals.cibr.common import plot_subject_spectra_separate
from signals.cibr.common import get_subject_spectra
from signals.cibr.common import get_mean_spectra
from signals.cibr.common import get_rest_intervals
from signals.cibr.common import get_peak_by_correlation


def plot_topomaps(save_path, mixing, raw_info, page, component_idxs):

    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # extract topomaps from the mixing matrix
    topomaps = np.abs(mixing)

    for ch_type in ['grad', 'planar1', 'planar2']:

        picks = mne.pick_types(raw_info, meg=ch_type)
        if ch_type == 'grad':
            pos = raw_info
        else:
            pos = mne.channels.layout._find_topomap_coords(raw_info, picks)

        # plot topomaps of selected (ordered) component indices
        fig_ = plt.figure()
        for i, idx in enumerate(component_idxs):
            topo_data = topomaps[picks, idx]
            axes = fig_.add_subplot(page, (len(component_idxs) - 1) / page + 1, i+1)
            mne.viz.plot_topomap(topo_data, pos, axes=axes, show=False)

        # save plotted maps
        if save_path:
            fig_.savefig(os.path.join(save_path, ch_type + '_topo.png'), dpi=310)



def save_spectrum_data(save_path, data, freqs, 
                     raw_times, component_idxs, names,
                     total_ivals):

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    subject_spectra = get_subject_spectra(data, freqs, 
        total_ivals, raw_times, normalized=False, 
        subtract_power_law=False)

    # prepare header
    header = ['Subject']
    header += list(freqs)
        
    data_array = []
    for idx, comp_idx in enumerate(component_idxs):
        for sub_idx in range(subject_spectra.shape[1]):
            row = [names[sub_idx] + ' (' + str(idx+1) + ')']
            
            for val in subject_spectra[comp_idx, sub_idx]:
                row.append(val)

            data_array.append(row)

    with open(os.path.join(save_path, 'spectrum_data.csv'),'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        for line in data_array:
            writer.writerow(line)


def save_peak_values(save_path, data, freqs, 
                     raw_times, component_idxs, correlations, names,
                     total_ivals):

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # extract mean spectra
    mean_spectra = get_mean_spectra(data, freqs)

    # extract spectra for different subjects
    subject_spectra_normalized = get_subject_spectra(data, freqs, 
        total_ivals, raw_times, normalized=True)

    total_subject_spectra_ws = get_subject_spectra(data, freqs, 
        total_ivals, raw_times, normalized=False, 
        subtract_power_law=False)

    # prepare header
    header = ['Subject']
    for i in range(len(component_idxs)):
        header.append('Frequency (' + str(i+1) + ')')
        header.append('Total amplitude (' + str(i+1) + ')')
        header.append('Variance (' + str(i+1) + ')')
        header.append('Score (' + str(i+1) + ')')
        
    data_array = []
    for sub_idx in range(subject_spectra_normalized.shape[1]):
        row = [names[sub_idx]]
        for comp_idx in component_idxs:
            psd_normalized = subject_spectra_normalized[comp_idx, sub_idx]
            peak_idx = get_peak_by_correlation(psd_normalized, mean_spectra[comp_idx])

            freq = freqs[peak_idx]
            total_peak_ws = total_subject_spectra_ws[comp_idx, sub_idx][peak_idx]  # noqa

            row.append(freq)
            row.append(total_peak_ws)
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
    parser.add_argument('--save_path')
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
    lpass = 30
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

    # raws = []
    # index_count = 0
    # total_ivals = []
    # for ival in ec_ivals:
    #     total_ival = (index_count, index_count + ival[1] - ival[0])
    #     index_count += ival[1] - ival[0]
    #     total_ivals.append(total_ival)
    #     raws.append(mne.io.RawArray(raw._data[:, ival[0]:ival[1]], raw.info))

    # raw = mne.concatenate_raws(raws)
    # print total_ivals

    freqs, times, orig_data, _ = calculate_stft(raw._data, sfreq, window_in_samples,
        overlap_in_samples, hpass, lpass, row_wise=True)

    shape, orig_data = orig_data.shape, arrange_as_matrix(orig_data)

    # data, mixing, dewhitening, _, _, mean = complex_ica(
    #     orig_data, n_components, conveps=conveps, maxiter=maxiter)

    ica = FastICA(
        n_components=n_components,
        algorithm='parallel',
        whiten=True,
        max_iter=10000,
        tol=0.000000001)

    data = ica.fit_transform(np.abs(orig_data).T).T
    mixing = ica.mixing_

    # back_proj = np.dot(mixing, data)
    # var_explained = (100 - 100*np.mean(np.var(orig_data - back_proj))/
    #                  np.mean(np.var(orig_data)))
    # del back_proj
    # del orig_data
    # print "Variance explained by components: " + str(var_explained)

    data = arrange_as_tensor(data, shape)

    # print "Calculating correlations."
    # correlations = get_correlations(data, freqs, total_ivals, raw.times)
    # corr_scores = np.sum(correlations, axis=1)

    # sort in reverse order
    # corr_idxs = np.argsort(-corr_scores)

    # print "Corr scores sorted: "
    # for idx in corr_idxs:
    #     print 'Component ' + str(idx+1) + ': ' + str(corr_scores[idx])

    print "Plotting brainmaps."
    plot_topomaps(cli_args.save_path, mixing, raw.info,
                   page, range(data.shape[0]))

    print "Plotting mean spectra."
    plot_mean_spectra(cli_args.save_path, data, freqs, page, range(data.shape[0]))

    print "Plotting subject spectra."
    plot_subject_spectra(cli_args.save_path, data, freqs, page, 
                         total_ivals, raw.times, range(data.shape[0]))

    # print "Plotting subject spectra to separate images"
    # plot_subject_spectra_separate(cli_args.save_path, data, freqs, page, 
    #                               total_ivals, raw.times, range(data.shape[0]),
    #                               names)

    # print "Saving to data values."
    # save all the values in same order as the plot
    # save_peak_values(cli_args.save_path, data, freqs,
    #                  raw.times, corr_idxs, correlations, names,
    #                  total_ivals)

    print "Saving spectrum data"
    save_spectrum_data(cli_args.save_path, data, freqs, 
                       raw.times, range(data.shape[0]), names, total_ivals)
