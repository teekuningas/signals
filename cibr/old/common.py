import os
import mne
import numpy as np

import scipy.signal
from scipy import stats

import matplotlib.pyplot as plt


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


def get_peak(freqs, spectrum, band=(6, 14)):
    nearby_freqs = np.where((freqs > band[0]) & (freqs < band[1]))[0]
    argmax_ = (nearby_freqs[0] + 
               np.argmax(scipy.signal.detrend(spectrum[nearby_freqs])))
    return freqs[argmax_], spectrum[argmax_]


def get_power(freqs, spectrum, band=(6,14)):
    band = np.where((freqs > band[0]) & (freqs < band[1]))[0]
    power = np.trapz(spectrum[band], freqs[band])
    return power


def load_raw(fnames):
    print "Loading data"
    if type(fnames) is not list:
        fnames = [fnames]
    raws = []
    for fname in fnames:
        raws.append(mne.io.Raw(fname, preload=True))

    raw = mne.concatenate_raws(raws)
    return raw


def preprocess(raw, filter_=(2, 35), min_duration=2):
    print "Preprocessing."

    events = mne.find_events(raw, shortest_event=1, min_duration=min_duration/raw.info['sfreq'], uint_cast=True)
    picks = mne.pick_types(raw.info, meg='grad')
    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names']) 
                       if idx not in picks])

    if filter_:
        raw.filter(l_freq=filter_[0], h_freq=filter_[1], verbose='error')

    return raw, events


def calculate_stft(data, sfreq, window, noverlap, hpass, lpass, row_wise=False):
    print "Calculating stft."

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

def calculate_istft(data, sfreq, window, noverlap, freq_pads):

    data_padded = np.pad(data, [(0,0), freq_pads, (0,0)], 
                          mode='constant')

    timeseries = scipy.signal.istft(data_padded, fs=sfreq, 
        nperseg=window, noverlap=noverlap)[1]

    return timeseries

def arrange_as_matrix(tensor):
    print "Arranging as matrix"
    fts = [tensor[:, :, idx] for idx in range(tensor.shape[2])]
    return np.concatenate(fts, axis=1)


def arrange_as_tensor(mat, shape):
    
    parts = np.split(mat, shape[2], axis=1)

    xw = mat.shape[0]
    yw = mat.shape[1]/shape[2]
    zw = shape[2]

    tensor = np.empty((xw, yw, zw), dtype=mat.dtype)
    for idx, part in enumerate(parts):
        tensor[:, :, idx] = part

    return tensor


def compute_stft_intervals(stft, events, first_samp, raw_length):

    intervals = {}
    for idx, event in enumerate(events):
        if event[2] not in intervals:
            intervals[event[2]] = []

        smin = int(((event[0] - first_samp) / float(raw_length)) * 
                   stft.shape[2] + 0.5)

        if idx == len(events) - 1:
            smax = stft.shape[2] - 1
        else:
            smax = int(((events[idx+1][0] - first_samp) / 
                        float(raw_length)) * stft.shape[2] + 0.5)

        intervals[event[2]].append((smin, smax))

    return intervals


def combine_stft_intervals(stft, intervals):
    parts = []
    for smin, smax in intervals:
        parts.append(stft[:, :, smin:smax])

    return np.concatenate(parts, axis=-1)


def compute_raw_intervals(raw, events):

    intervals = {}
    for idx, event in enumerate(events):
        if event[2] not in intervals:
            intervals[event[2]] = []

        if idx == len(events) - 1:
            tmax = raw._data.shape[1] - 1
        else:
            tmax = events[idx+1][0] - raw.first_samp

        tmin = event[0] - raw.first_samp

        # convert to seconds
        tmin = tmin / raw.info['sfreq']
        tmax = tmax / raw.info['sfreq']

        intervals[event[2]].append((tmin, tmax))

    return intervals


def combine_raw_intervals(raw, intervals):
    raws = []
    for interval in intervals:
        data = raw.copy().crop(tmin=interval[0], tmax=interval[1])._data
        raws.append(mne.io.RawArray(data, raw.info))

    return mne.concatenate_raws(raws)


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
    for i in range(subject_spectra.shape[0]):
        correlations_i = []
        for j in range(subject_spectra.shape[1]):
            ts1 = mean_spectra[i]
            ts2 = subject_spectra[i, j]

            ts2 = roll_subject_to_average(ts2, ts1)
 
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
                         dpi=310)


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

