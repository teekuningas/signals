import sys
import argparse

import mne
import numpy as np
import matplotlib.pyplot as plt 

import scipy.signal

from signals.ica.complex_ica import complex_ica


def load_raw(fnames):
    print "Loading data"
    raws = []
    for fname in fnames:
        raws.append(mne.io.Raw(fname, preload=True))

    raw = mne.concatenate_raws(raws)
    return raw


def preprocess(raw):
    print "Preprocessing."

    events = mne.find_events(raw)
    picks = mne.pick_types(raw.info, meg='grad')
    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names']) 
                       if idx not in picks])

    raw.filter(l_freq=2, h_freq=40)

    return raw, events


def combine_raw_intervals(raw, intervals):
    raws = []
    for interval in intervals:
        data = raw.copy().crop(tmin=interval[0], tmax=interval[1])._data
        raws.append(mne.io.RawArray(data, raw.info))

    return mne.concatenate_raws(raws)


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


def compute_src_intervals():
    pass


def combine_src_intervals():
    pass


def cortical_projection(raw, fwd_fname, cov_fname):
    print "Projeting to cortical space."
    
    fwd = mne.forward.read_forward_solution(fwd_fname)
    cov = mne.compute_raw_covariance(mne.io.Raw(cov_fname, preload=True))
    inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov)
    stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=0.1)

    return stc


def calculate_stft(data, sfreq, window, noverlap, hpass, lpass):
    print "Calculating stft."
    freqs, times = scipy.signal.stft(data[0], fs=sfreq, nperseg=window, 
                                     noverlap=noverlap)[:2]

    ch_stfts = []
    for ch_data in data:
        ch_stfts.append(
            scipy.signal.stft(ch_data, fs=sfreq, 
                              nperseg=window, noverlap=noverlap)[2])

    stft = np.array(ch_stfts)

    hpass_ind = min(np.where(freqs >= hpass)[0])
    lpass_ind = max(np.where(freqs <= lpass)[0])

    freqs = freqs[hpass_ind:lpass_ind]

    stft = stft[:, hpass_ind:lpass_ind, :]

    return freqs, times, stft


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


def visualize_components(data, mixing, dewhitening, freqs, vertices):
    brainmaps = np.abs(np.dot(dewhitening, mixing))
    spectrums = np.mean(np.power(np.abs(data), 2), axis=2)

    # plot all spectrums
    fig_ = plt.figure()
    page = 10
    for i in range(spectrums.shape[0]):
        y = spectrums[i]
        x = freqs
        axes = fig_.add_subplot(page, (spectrums.shape[0] - 1) / page + 1, i+1)
        axes.plot(x,y)

    plt.show()

    brainmap_stc = mne.SourceEstimate(brainmaps[:, 0, np.newaxis], vertices=vertices, tmin=0, tstep=1)


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
    n_components = 15
    conveps = 1e-7
    maxiter = 8000
    hpass = 4
    lpass = 25
    window_in_samples = np.power(2, np.ceil(np.log(
        sfreq * window_in_seconds)/np.log(2)))
    overlap_in_samples = (window_in_samples * 3) / 4

    raw, events = preprocess(raw)

    intervals = compute_raw_intervals(raw, events)
    meditation_raw = combine_raw_intervals(raw, intervals[10][:2])

    stc = cortical_projection(meditation_raw, cli_args.fwd, cli_args.empty)

    freqs, times, data = calculate_stft(stc.data, sfreq, window_in_samples, 
                                        overlap_in_samples, hpass, lpass)

    shape, data = data.shape, arrange_as_matrix(data)

    data, mixing, dewhitening, _, _, mean = complex_ica(
        data, n_components, conveps=conveps, maxiter=maxiter)

    data = arrange_as_tensor(data, shape)

    visualize_components(data, mixing, dewhitening, freqs, stc.vertices)

    import pdb; pdb.set_trace()
    print "miau"


