import mne

import argparse
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import hilbert
from scipy.signal import decimate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raws', nargs='+')
    parser.add_argument('--empty', nargs='+')
    parser.add_argument('--save-path')

    cli_args = parser.parse_args()

    raws = []
    for fname in sorted(cli_args.raws):

        raw = mne.io.Raw(fname, preload=True)
        raw.resample(50)

        grads = mne.pick_types(raw.info, meg='grad')
        raw.drop_channels([ch_name for idx, ch_name in 
                           enumerate(raw.info['ch_names']) 
                           if idx not in grads])
        raws.append(raw)

    n_subjects = len(raws)
    n_rows = int(np.sqrt(n_subjects))
    n_cols = ncols=int(np.ceil((n_subjects/float(n_rows))))

#     energies = []
#     for raw in raws:
#         energies.append(np.mean(raw._data**2, axis=1))
#     energies = np.array(energies)
# 
#     vmin, vmax = 0, 0
#     for subj_energy in energies:
#         if vmin > np.min(subj_energy):
#             vmin = np.min(subj_energy)
#         if vmax < np.max(subj_energy):
#             vmax = np.max(subj_energy)
# 
#     fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
#     for subj_idx, subj_energy in enumerate(energies):
#         row_idx, col_idx = subj_idx/n_cols, subj_idx%n_cols
#         ax = axes[row_idx, col_idx]
#         mne.viz.plot_topomap(subj_energy, raw.info, axes=ax, show=False, 
#                              vmin=vmin, vmax=vmax)
# 
#     print "Computing hilbert transforms"
#     data = np.concatenate([np.abs(hilbert(raw._data)) for raw in raws], axis=1)
# 
#     areas = ['Left-temporal', 'Right-temporal', 'Left-occipital', 'Right-occipital']
#     average_data = []
#     for area in areas:
#         selected_ch_names = mne.utils._clean_names(
#             mne.read_selection(area),
#             remove_whitespace=True)
#         ch_idxs = [ch_idx for ch_idx, ch_name in enumerate(raw.info['ch_names']) if
#                    ch_name in selected_ch_names]
#         component = np.mean(data[ch_idxs], axis=0)
#         average_data.append(component)
#     average_data = np.array(average_data)
# 
#     fig, axes = plt.subplots(average_data.shape[0])
#     for idx, ax in enumerate(axes):
#         ax.plot(average_data[idx])
# 
#     print "Computing hilbert transforms"
#     data = np.concatenate([np.abs(hilbert(raw.copy().filter(l_freq=6, h_freq=14)._data)) for raw in raws], axis=1)
# 
#     areas = ['Left-temporal', 'Right-temporal', 'Left-occipital', 'Right-occipital']
#     average_data = []
#     for area in areas:
#         selected_ch_names = mne.utils._clean_names(
#             mne.read_selection(area),
#             remove_whitespace=True)
#         ch_idxs = [ch_idx for ch_idx, ch_name in enumerate(raw.info['ch_names']) if
#                    ch_name in selected_ch_names]
#         component = np.mean(data[ch_idxs], axis=0)
#         average_data.append(component)
#     average_data = np.array(average_data)
# 
#     fig, axes = plt.subplots(average_data.shape[0])
#     for idx, ax in enumerate(axes):
#         ax.plot(average_data[idx])


    print "Computing hilbert transforms"
    data = np.concatenate([np.abs(hilbert(raw.copy().filter(l_freq=6, h_freq=14)._data[:, ::100])) for raw in raws], axis=1)
    # data = np.concatenate([np.abs(hilbert(raw.copy()._data[:, ::100])) for raw in raws], axis=1)

    areas = ['Left-temporal', 'Right-temporal', 'Left-occipital', 'Right-occipital']
    average_data = []
    for area in areas:
        selected_ch_names = mne.utils._clean_names(
            mne.read_selection(area),
            remove_whitespace=True)
        ch_idxs = [ch_idx for ch_idx, ch_name in enumerate(raw.info['ch_names']) if
                   ch_name in selected_ch_names]
        component = np.mean(data[ch_idxs], axis=0)
        average_data.append(component)
    average_data = np.array(average_data)

    fig, axes = plt.subplots(average_data.shape[0])
    for idx, ax in enumerate(axes):
        ax.plot(average_data[idx], linewidth=0.1)
        ax.set_ylim(np.percentile(average_data[idx], 0.5), np.percentile(average_data[idx], 99.5))

    plt.show()

    import pdb; pdb.set_trace()
    print("miau")


