import argparse
import os
import sys

from pprint import pprint

import matplotlib
matplotlib.use('Agg')
matplotlib.rc('figure', max_open_warning=0)

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import mne
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA


from fooof import FOOOFGroup

from signals.cibr.lib.stc import plot_vol_stc_brainmap


def save_fig(fig, path, dpi):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig.savefig(path, dpi=dpi)

def arrange_as_matrix(tensor):
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


parser = argparse.ArgumentParser()
parser.add_argument('--raw', nargs='+')
parser.add_argument('--med_recon')
parser.add_argument('--ic_recon')
parser.add_argument('--empty', nargs='+')
parser.add_argument('--save_path')

cli_args = parser.parse_args()

prefilter_band = (1, 40)
sampling_rate = 100
vol_spacing = '8'
n_components = 40
random_state = 10

empty_paths = cli_args.empty
empty_raws = []
for path in empty_paths:
    raw = mne.io.read_raw_fif(path, preload=True, verbose='error')
    raw.filter(*prefilter_band)
    raw.crop(tmin=(raw.times[0]+3), tmax=raw.times[-1]-3)
    raw.resample(sampling_rate)
    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names'])
                       if idx not in mne.pick_types(raw.info, meg=True)])
    empty_raws.append(raw)
empty_raw = mne.concatenate_raws(empty_raws)
empty_raws = []

noise_cov = mne.compute_raw_covariance(empty_raw, method='empirical')

stfts = []
names = []
for path in cli_args.raw:

    raw = mne.io.read_raw_fif(path, preload=True)

    raw.filter(*prefilter_band)
    raw.resample(sampling_rate)
    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names'])
                       if idx not in mne.pick_types(raw.info, meg=True)])

    fname = os.path.basename(path)
    folder = os.path.dirname(path)

    if fname.startswith('subject_'):
        subjects_dir = cli_args.med_recon
        subject = fname.split('_rest')[0]
        trans = os.path.join(folder, subject + '-trans.fif')
    elif fname.startswith('IC_'):
        subjects_dir = cli_args.ic_recon
        subject = 'IC_' + fname.split('_')[2][1:]
        trans = os.path.join(folder, subject + '-trans.fif')
    else:
        raise Exception('Not implemented')

    names.append(subject)

    bem = os.path.join(subjects_dir, subject, 'bem',
                       subject+'-inner_skull-bem-sol.fif')
    src_fname = os.path.join(subjects_dir, subject, 'bem',
                             subject + '-vol-' + vol_spacing + '-src.fif')
    src = mne.source_space.read_source_spaces(src_fname)

    print("Creating forward solution..")
    fwd = mne.make_forward_solution(
        info=raw.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        verbose='warning')

    print("Creating inverse operator..")
    inv = mne.minimum_norm.make_inverse_operator(
        info=raw.info,
        forward=fwd,
        noise_cov=noise_cov,
        depth=None,
        fixed=False,
        verbose='warning')

    # tmin = round(raw.times[-1]*0.55)
    # tmax = round(raw.times[-1]*0.95)

    tmin = round(raw.times[-1]*0.60)
    tmax = round(raw.times[-1]*0.90)

    # ticks = np.arange(tmin, tmax, 1)
    ticks = np.arange(tmin, tmax, 2)

    stft = []
    for tick_idx in range(len(ticks)-2):
        # ival = ticks[tick_idx], ticks[tick_idx+1]
        ival = ticks[tick_idx], ticks[tick_idx+2]
        # n_fft = int(raw.info['sfreq']*2)
        n_fft = int(raw.info['sfreq']*4)

        raw_array = raw.copy().crop(ival[0], ival[1])
        print("Computing psd for ival " + str(ival[0]) + ' - ' + str(ival[1]))
        psd = mne.minimum_norm.compute_source_psd(
            raw_array, inv, method='dSPM', n_fft=n_fft, pick_ori=None, dB=False,
            prepared=False, fmin=1, fmax=24, verbose='warning')
        freqs = psd.times
        vertices = psd.vertices[0]
        stft.append(psd.data)

    stft = np.rollaxis(np.array(stft), axis=0, start=3)

    stfts.append(stft)

means = []
stds = []
for stft in stfts:
    # means.append(np.mean(stft, axis=(1,2))[:, np.newaxis, np.newaxis])
    # stds.append(np.std(stft, axis=(1,2))[:, np.newaxis, np.newaxis])
    mean_ = np.mean(stft, axis=2)[:, :, np.newaxis]
    std_ = np.std(stft - mean_, axis=(1,2))[:, np.newaxis, np.newaxis]
    means.append(mean_)
    stds.append(std_)

# allocate space for the concatenated array
data = np.zeros((stfts[0].shape[0], sum([stft.shape[1]*stft.shape[2] for stft in stfts])))

print("Create concatenated dataset with frequencies and times flattened to frequency-time")
concat_idx = 0
for stft_idx, stft in enumerate(stfts):

    # stft = stft - np.mean(means, axis=0)
    # stft = (stft - means[stft_idx]) / stds[stft_idx]
    stft = (stft - means[stft_idx]) / stds[stft_idx]

    data[:, concat_idx:(concat_idx + stft.shape[1]*stft.shape[2])] = arrange_as_matrix(stft)
    concat_idx += stft.shape[1]*stft.shape[2]

import pdb; pdb.set_trace()

pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
data = pca.fit_transform(data.T).T
print("Explained variance: " + str(np.sum(pca.explained_variance_ratio_)))

print("Running ICA")
ica = FastICA(whiten=False, random_state=random_state)
data = ica.fit_transform(data.T).T

print("Recreate subject stft sources from concatenated sources")
sources_stft = []
tidx = 0
for subject_idx in range(len(stfts)):
    length = stfts[subject_idx].shape[1]*stfts[subject_idx].shape[2]
    sources_stft.append(arrange_as_tensor(data[:, tidx:tidx+length], 
                                          stfts[subject_idx].shape))
    tidx += length

unmixing = ica.components_ @ pca.components_
ordering = np.argsort([-np.sum(unmixing[idx]**2) for idx in range(n_components)])

# spectrums = np.rollaxis(np.array([np.sqrt(np.sum(stft**2, axis=2)) for stft in sources_stft]), axis=1)[ordering]
spectrums = np.rollaxis(np.array([np.var(stft, axis=2) for stft in sources_stft]), axis=1)[ordering]

spatial_maps = (np.sqrt(unmixing**2))[ordering]
grand_spectrums = np.mean(spectrums, axis=1)

# select specific components
component_idxs = np.array([32, 10, 22, 36, 5, 39, 7, 18]) - 1
spectrums = spectrums[component_idxs]
spatial_maps = spatial_maps[component_idxs]
grand_spectrums = grand_spectrums[component_idxs]
n_components = len(component_idxs)

fooofs = []
csv_data_freqs = []
csv_data_amps = []
for comp_idx in range(n_components):
    print("FOOOF for component " + str(comp_idx+1) + ": ")
    fg = FOOOFGroup(max_n_peaks=4, peak_threshold=1.0);
    fg.fit(freqs, spectrums[comp_idx])

    fooofs.append(fg)

    peaks = []
    for subj_idx in range(len(names)):
        fooof_result = fg.group_results[subj_idx]
        best_peak_idx = None
        for peak_idx, peak in enumerate(fooof_result.peak_params):
            if np.abs(peak[0] - 10) < 3.0:
                if best_peak_idx is None:
                    best_peak_idx = peak_idx
                    continue
                if fooof_result.peak_params[best_peak_idx][1] < peak[1]:
                    best_peak_idx = peak_idx
        if best_peak_idx is not None:
            peaks.append(fooof_result.peak_params[best_peak_idx])
        else:
            peaks.append(None)

    comp_freqs = []
    mean_freq = np.mean([peak[0] for peak in peaks if peak is not None])
    for subj_idx in range(len(names)):
        if peaks[subj_idx] is not None:
            comp_freqs.append(peaks[subj_idx][0])
        else:
            comp_freqs.append(mean_freq)
    csv_data_freqs.append(comp_freqs)

    comp_amps = []
    mean_amp = np.mean([peak[1] for peak in peaks if peak is not None])
    for subj_idx in range(len(names)):
        if peaks[subj_idx] is not None:
            comp_amps.append(peaks[subj_idx][1])
        else:
            comp_amps.append(mean_amp)
    csv_data_amps.append(comp_amps)

    pprint(peaks)

if not os.path.exists(cli_args.save_path):
    os.makedirs(cli_args.save_path)

with open(os.path.join(cli_args.save_path, 'freqs.csv'), 'w') as f:
    f.write(', ' + ', '.join(['Component ' + str(idx+1).zfill(2) for idx in range(n_components)]) +
            '\n')
    for row_idx, row in enumerate(np.array(csv_data_freqs).T):
        f.write(names[row_idx] + ', ' + ', '.join([str(round(elem, 3)) for elem in row]) + '\n')

if not os.path.exists(cli_args.save_path):
    os.makedirs(cli_args.save_path)
with open(os.path.join(cli_args.save_path, 'amps.csv'), 'w') as f:
    f.write(', ' + ', '.join(['Component ' + str(idx+1).zfill(2) for idx in range(n_components)]) +
            '\n')
    for row_idx, row in enumerate(np.array(csv_data_amps).T):
        f.write(names[row_idx] + ', ' + ', '.join([str(round(elem, 3)) for elem in row]) + '\n')

n_cols = 4
n_rows = int((n_components - 1) / n_cols) + 1

print("Plot spectrums")
fig, axes = plt.subplots(n_rows, n_cols)
fig.set_size_inches(n_cols*7, n_rows*5)
axes = [item for sublist in axes for item in sublist]
for comp_idx in range(n_components):
    ax = axes[comp_idx]
    ax.set_title("Component " + str(comp_idx+1))
    for subj_idx in range(len(stfts)):
        ax.plot(freqs, spectrums[comp_idx, subj_idx])
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
fig.tight_layout()
save_fig(fig, os.path.join(cli_args.save_path, 'spectrums.png'), 100)

print("Plot grand spectrums")
fig, axes = plt.subplots(n_rows, n_cols)
fig.set_size_inches(n_cols*7, n_rows*5)
axes = [item for sublist in axes for item in sublist]
for comp_idx in range(n_components):
    ax = axes[comp_idx]
    ax.set_title("Component " + str(comp_idx+1))
    ax.plot(freqs, grand_spectrums[comp_idx])
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
fig.tight_layout()
save_fig(fig, os.path.join(cli_args.save_path, 'grand_spectrums.png'), 100)

print("Plot individual spectrums")
for subject_idx in range(spectrums.shape[1]):
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(n_cols*7, n_rows*5)
    axes = [item for sublist in axes for item in sublist]
    for comp_idx in range(n_components):
        ax = axes[comp_idx]
        ax.set_title("Component " + str(comp_idx+1))
        ax.plot(freqs, spectrums[comp_idx, subject_idx, :])
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power')
    fig.tight_layout()
    fname = 'spectrums_' + str(subject_idx+1).zfill(2) + '_' + names[subject_idx] + '.png'
    save_fig(fig, os.path.join(cli_args.save_path, 'individuals', fname), 100)


print("Plot FOOOF fits")
for subject_idx in range(len(names)):
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(n_cols*7, n_rows*5)
    axes = [item for sublist in axes for item in sublist]
    for comp_idx in range(n_components):
        ax = axes[comp_idx]
        fm = fooofs[comp_idx].get_fooof(subject_idx)
        try:
            fm.plot(ax=ax, plot_peaks='dot')
        except:
            print("Not plotting, no peaks")
        ax.set_title(str(comp_idx+1).zfill(2) + ' (' + str(fm.r_squared_) + ')')
     
    fig.tight_layout()
    fname = 'fooof_' + str(subject_idx+1).zfill(2) + '_' + names[subject_idx] + '.png'
    save_fig(fig, os.path.join(cli_args.save_path, 'fooofs', fname), 100)


print("Plot spatial maps")
for comp_idx, spatial_map in enumerate(spatial_maps):
    fig, ax = plt.subplots()
    plot_vol_stc_brainmap(spatial_map, vertices, vol_spacing, subjects_dir, ax, cap=0.0)
    save_fig(fig, os.path.join(cli_args.save_path, 'spatial_maps', 
                               'map_' + str(comp_idx+1).zfill(2) + '.png'), 100)

