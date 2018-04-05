import sys

import argparse

import mne
import numpy as np

import matplotlib.pyplot as plt

from mne.channels.layout import find_layout
from mne.channels.layout import _pair_grad_sensors

from scipy.signal import detrend

from signal.common import get_power
from signal.common import get_peak

import matplotlib
matplotlib.rc('font', size=6)

STATES = {
    'rest': {
        'id': 11
    },
    'mind': {
        'id': 10
    },
    'anx': {
        'id': 13
    },
    'plan': {
        'id': 12
    }
}

def preprocess(paths):

    print "Processing files: "
    for path in paths:
        print path

    raws = []
    for path in paths:
        raw = mne.io.Raw(path, preload=True, verbose='error')

        raws.append(raw)

    raw = mne.concatenate_raws(raws)


    try:
        events = mne.find_events(raw, min_duration=2/raw.info['sfreq'],
                                 verbose='error')
    except:
        import pdb; pdb.set_trace()
        events = mne.find_events(raw, shortest_event=1)

    # select only gradiometer channels
    picks = mne.pick_types(raw.info, meg='grad')
    raw.drop_channels([ch_name for idx, ch_name in enumerate(raw.info['ch_names'])
                       if idx not in picks]) 

    raw.drop_channels(raw.info['bads'])

    raw._data = (raw._data - np.mean(raw._data)) / np.std(raw._data)

    return raw, events

def get_state_data(raw, events):

    data = raw._data

    # collect different states from data

    state_data = {}

    wsize_factor = 1
    wsize = np.power(2, np.ceil(np.log(raw.info['sfreq'] * wsize_factor)/np.log(2)))

    for key, state in STATES.items():
        intervals = []
        for idx, event in enumerate(events):
            if event[2] == state['id']:
                tmin = (event[0] - raw.first_samp) / raw.info['sfreq']

                if idx == len(events) - 1:
                    tmax = tmin + 110
                    if event[2] != 11:
                        continue
                else:
                    tmax = (events[idx+1][0] - raw.first_samp) / raw.info['sfreq']

                intervals.append((tmin, tmax))

        temp_psds = []
        for tmin, tmax in intervals:
            psds, freqs = mne.time_frequency.psd_welch(raw, fmin=4, fmax=16, tmin=tmin, tmax=tmax, n_fft=wsize)
            temp_psds.append(psds)

        temp_psds = np.array(temp_psds)
        averaged = np.mean(temp_psds, axis=0)
        
        state_data[key] = averaged

    return freqs, state_data


def plot_topography(raw, events):

    freqs, state_data = get_state_data(raw, events)

    def subplot_callback(ax, ch_idx):
        """
        """
        ax.plot(freqs, state_data['mind'][ch_idx], color='blue')
        ax.plot(freqs, state_data['anx'][ch_idx], color='cyan')
        ax.plot(freqs, state_data['plan'][ch_idx], color='yellow')
        ax.plot(freqs, state_data['rest'][ch_idx], color='red')

        mind_argmax, _ = get_peak(freqs, state_data['mind'][ch_idx])
        plan_argmax, _ = get_peak(freqs, state_data['plan'][ch_idx])
        anx_argmax, _ = get_peak(freqs, state_data['anx'][ch_idx])
        rest_argmax, _ = get_peak(freqs, state_data['rest'][ch_idx])

        ax.axvline(mind_argmax)
        ax.axvline(plan_argmax)
        ax.axvline(anx_argmax)

        ax.set_xlabel = 'Frequency (Hz)'
        ax.set_ylabel = 'Power (dB)'
        plt.show()

    for ax, idx in mne.viz.iter_topography(raw.info,
                                           fig_facecolor='black',
                                           axis_facecolor='black',
                                           axis_spinecolor='black',
                                           on_pick=subplot_callback):
        ax.plot(freqs, state_data['mind'][idx], color='blue')
        ax.plot(freqs, state_data['anx'][idx], color='cyan')
        ax.plot(freqs, state_data['plan'][idx], color='yellow')
        ax.plot(freqs, state_data['rest'][idx], color='red')

        mind_argmax, _ = get_peak(freqs, state_data['mind'][idx])
        plan_argmax, _ = get_peak(freqs, state_data['plan'][idx])
        anx_argmax, _ = get_peak(freqs, state_data['anx'][idx])
        rest_argmax, _ = get_peak(freqs, state_data['rest'][idx])

        ax.axvline(mind_argmax)
        ax.axvline(plan_argmax)
        ax.axvline(anx_argmax)
        ax.axvline(rest_argmax)

    plt.show(block=False)
    plt.pause(5)


def save_data(raw, events, type_, output_folder='', fname=''):

    freqs, state_data = get_state_data(raw, events)

    topomap_data = np.zeros(state_data[type_].shape[0])
    for i in range(topomap_data.shape[0]):
        _, peak = get_peak(freqs, state_data[type_][i])
        topomap_data[i] = peak

    # cant trust plot_topomap behavior here as we dont have time series data
    def _merge_grad_data(data):
        """
        """
        new_data = np.zeros(data.shape[0] / 2)
        for i in range(data.shape[0]/2):
            new_data[i] = (data[2*i] + data[2*i+1]) / 2.0
        return new_data
    picks, pos = _pair_grad_sensors(raw.info, find_layout(raw.info))
    topomap_data = _merge_grad_data(topomap_data[picks]).reshape(-1)

    if output_folder and fname:
        topo_name = "".join([output_folder, fname, "_", type_, ".csv"])
        np.savetxt(topo_name, topomap_data)


def save_power_data(raw, events, type_, output_folder='', fname=''):

    freqs, state_data = get_state_data(raw, events)

    topomap_data = np.zeros(state_data[type_].shape[0])
    for i in range(topomap_data.shape[0]):
        power = get_power(freqs, state_data[type_][i])
        topomap_data[i] = power

    # cant trust plot_topomap behavior here as we dont have time series data
    def _merge_grad_data(data):
        """
        """
        new_data = np.zeros(data.shape[0] / 2)
        for i in range(data.shape[0]/2):
            new_data[i] = (data[2*i] + data[2*i+1]) / 2.0
        return new_data

    picks, pos = _pair_grad_sensors(raw.info, find_layout(raw.info))
    topomap_data = _merge_grad_data(topomap_data[picks]).reshape(-1)

    if output_folder and fname:
        topo_name = "".join([output_folder, fname, "_", type_, "_power.csv"])
        np.savetxt(topo_name, topomap_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('output_folder')
    parser.add_argument('input_file', nargs='+')
    cli_args = parser.parse_args()
    output_folder = cli_args.output_folder

    fname = cli_args.input_file[0].split('/')[-1].split('.fif')[0]

    raw, events = preprocess(cli_args.input_file)
    # plot_topography(raw, events)

    save_data(raw, events, 'mind', output_folder, fname)
    save_data(raw, events, 'plan', output_folder, fname)
    save_data(raw, events, 'anx', output_folder, fname)
    save_data(raw, events, 'rest', output_folder, fname)

    save_power_data(raw, events, 'mind', output_folder, fname)
    save_power_data(raw, events, 'plan', output_folder, fname)
    save_power_data(raw, events, 'anx', output_folder, fname)
    save_power_data(raw, events, 'rest', output_folder, fname)

