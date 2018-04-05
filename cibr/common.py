import mne
import numpy as np

import scipy.signal
from scipy import stats


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

    # return y without the power law
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


def preprocess(raw, filter_=True):
    print "Preprocessing."

    events = mne.find_events(raw, shortest_event=1, min_duration=2/raw.info['sfreq'])
    picks = mne.pick_types(raw.info, meg='grad')
    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names']) 
                       if idx not in picks])

    if filter_:
        raw.filter(l_freq=2, h_freq=35)

    return raw, events


def cortical_projection(raw, fwd, cov):
    print "Projeting to cortical space."
    
    inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov)
    stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=0.1)

    return stc


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

