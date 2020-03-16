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

from ica.complex_ica import complex_ica


def get_power_law(x, y):
    # replace infs
    mask = y == 0
    y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

    # then do log-log transform
    x_log = np.log10(x)
    y_log = np.log10(y)

    # estimate slope and intercept with linear regression
    slope, intercept, _, _, _ = stats.linregress(x_log, y_log)
    y_log_fitted = intercept + slope*x_log

    # transform the model back to original space
    y_fitted = 10**y_log_fitted

    return y_fitted



def calculate_stft(data, sfreq, window, noverlap, hpass, lpass, row_wise=False):
    print("Calculating stft.")

    rows = []
    if row_wise:
        for idx in range(data.shape[0]):
            freqs, times, row_stft = scipy.signal.stft(data[idx], 
                fs=sfreq, nperseg=window, noverlap=noverlap)
            rows.append(row_stft)
        stft = np.array(rows)

    else:
        freqs, times, stft = scipy.signal.stft(data, fs=sfreq, 
            nperseg=window, noverlap=noverlap)

    hpass_ind = min(np.where(freqs >= hpass)[0])
    lpass_ind = max(np.where(freqs <= lpass)[0])

    istft_freq_pads = (hpass_ind, len(freqs) - lpass_ind)

    freqs = freqs[hpass_ind:lpass_ind]

    stft = stft[:, hpass_ind:lpass_ind, :]

    return freqs, times, stft, istft_freq_pads

def arrange_as_matrix(tensor):
    print("Arranging as matrix")
    fts = [tensor[:, :, idx] for idx in range(tensor.shape[2])]
    return np.concatenate(fts, axis=1)


def arrange_as_tensor(mat, shape):
    
    parts = np.split(mat, shape[2], axis=1)

    xw = int(mat.shape[0])
    yw = int(mat.shape[1]/shape[2])
    zw = int(shape[2])

    tensor = np.empty((xw, yw, zw), dtype=mat.dtype)
    for idx, part in enumerate(parts):
        tensor[:, :, idx] = part

    return tensor


def get_mean_spectra(data, freqs):
    mean_spectra = []  
    for i in range(data.shape[0]):
        spectrum = np.mean(np.abs(data[i]), axis=-1)
        plaw = get_power_law(freqs, spectrum)
        psd = spectrum - plaw
        psd = (psd - np.min(psd)) / (np.max(psd) - np.min(psd))
        mean_spectra.append(psd)

    return np.array(mean_spectra)


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


def _roll_subject_to_average(subject_psd, average_psd):
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
    for i in range(subject_spectra.shape[0]):
        correlations_i = []
        for j in range(subject_spectra.shape[1]):
            ts1 = mean_spectra[i]
            ts2 = subject_spectra[i, j]

            ts2 = _roll_subject_to_average(ts2, ts1)
 
            r, _ = scipy.stats.pearsonr(ts1, ts2) 
            correlations_i.append(r)
        correlations.append(correlations_i)
    correlations = np.array(correlations)

    return correlations


def plot_topomaps(save_path, dewhitening, mixing, mean, raw_info, page, component_idxs):

    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # extract topomaps from the mixing matrix
    topomaps = np.abs(np.dot(dewhitening, mixing) + mean[:, np.newaxis])

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
            fig_.savefig(os.path.join(save_path, ch_type + '_topo.png'), 
                         dpi=620)


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
            spectrum = _roll_subject_to_average(psd, mean_spectra[idx])
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





def save_spectrum_data(save_path, data, freqs, 
                     raw_times, component_idxs, correlations, names,
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


def get_hiit_ec_intervals(splits_in_samples, sfreq):
    ivals = []
    for idx in range(len(splits_in_samples) - 1):
        subject_start = splits_in_samples[idx]
        subject_end = splits_in_samples[idx+1]
        ival_start = int(subject_start + 10*sfreq)
        ival_end = int((2/3.0)*(subject_end - subject_start) - 10*sfreq)
        print("Selecting interval " + str(ival_start) + ' - ' + str(ival_end) + 
              " for subject " + str(idx+1))
        ivals.append((ival_start, ival_end))
    return ivals



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

        # load raw
        raw = mne.io.Raw(path, preload=True)

        # keep only grads
        picks = mne.pick_types(raw.info, meg='grad')
        raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names'])
                           if idx not in picks])

        raw.resample(100)

        # raw._data = (raw._data - np.mean(raw._data)) / np.std(raw._data)

        raws.append(raw)

        names.append(raw.filenames[0].split('/')[-1].split('.fif')[0])

        splits_in_samples.append(splits_in_samples[-1] + len(raw))

    raw = mne.concatenate_raws(raws)

    sfreq = raw.info['sfreq']
    page = 10
    window_in_seconds = 2
    n_components = 20
    conveps = 1e-7
    maxiter = 15000
    hpass = 4
    lpass = 16
    window_in_samples = np.power(2, np.ceil(np.log(
        sfreq * window_in_seconds)/np.log(2)))
    overlap_in_samples = (window_in_samples * 3) / 4

    intervals = get_hiit_ec_intervals(splits_in_samples, sfreq)

    import pdb; pdb.set_trace()

    freqs, times, orig_data, _ = calculate_stft(raw._data, sfreq, window_in_samples,
        overlap_in_samples, hpass, lpass, row_wise=True)

    shape, orig_data = orig_data.shape, arrange_as_matrix(orig_data)

    data, mixing, dewhitening, _, _, mean = complex_ica(
        orig_data, n_components, conveps=conveps, maxiter=maxiter)

    back_proj = np.dot(np.dot(dewhitening, mixing), data) + mean[:, np.newaxis]
    var_explained = (100 - 100*np.mean(np.var(orig_data - back_proj))/
                     np.mean(np.var(orig_data)))
    del back_proj
    del orig_data
    print "Variance explained by components: " + str(var_explained)

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
    plot_topomaps(cli_args.save_path, dewhitening, mixing, mean, raw.info,
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
    save_peak_values(cli_args.save_path, data, freqs,
                     raw.times, corr_idxs, correlations, names,
                     total_ivals)

    print "Saving spectrum data"
    save_spectrum_data(cli_args.save_path, data, freqs, 
                       raw.times, corr_idxs, correlations, names, total_ivals)


