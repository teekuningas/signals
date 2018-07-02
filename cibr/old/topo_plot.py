import sys

import argparse

import mne
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
from scipy import stats


import matplotlib
matplotlib.rc('font', size=6)


def load_meditation_data(datasets):
    data = {
        'mind': [],
        'plan': [],
        'anx': [],
        'rest': [],
    }

    for idx, path in enumerate(datasets):
        print "Processing data: " + str(path)
        current = np.loadtxt(path)
        if 'mind' in path:
            data['mind'].append(current)
        elif 'plan' in path:
            data['plan'].append(current)
        elif 'anx' in path:
            data['anx'].append(current)
        elif 'rest' in path:
            data['rest'].append(current)

    data['mind'] = np.array(data['mind'])
    data['plan'] = np.array(data['plan'])
    data['anx'] = np.array(data['anx'])
    data['rest'] = np.array(data['rest'])

    return data


def load_combined_meditation_data(datasets):

    subjects = {}

    # load and group by subjects
    for idx, path in enumerate(datasets):
        print "Processing data: " + str(path)

        # subject_xxx_...
        prefix = path.split('/')[-1][0:11]

        if 'block1' in path:
            key = path.split('block1_')[0] + path.split('block1_')[1]

        elif 'block2' in path:
            key = path.split('block2_')[0] + path.split('block2_')[1]
        else:
            key = path

        if key in subjects:
            subjects[key].append(np.loadtxt(path))
        else:
            subjects[key] = [np.loadtxt(path)]

    # average blocks for subjects
    for key, blocks in subjects.items():
        subjects[key] = np.mean(subjects[key], axis=0)

    # then group by types
    data = {
        'mind': [],
        'plan': [],
        'anx': [],
        'rest': [],
    }

    for path, current in subjects.items():
        if 'mind' in path:
            data['mind'].append(current)
        elif 'plan' in path:
            data['plan'].append(current)
        elif 'anx' in path:
            data['anx'].append(current)
        elif 'rest' in path:
            data['rest'].append(current)

    data['mind'] = np.array(data['mind'])
    data['plan'] = np.array(data['plan'])
    data['anx'] = np.array(data['anx'])
    data['rest'] = np.array(data['rest'])

    return data


def plot_in_subplots(data, type_='', save_path=''):
    fig_ = plt.figure()
    for idx, topo in enumerate(data):
        axes = fig_.add_subplot(np.ceil(len(data)/4.0), 4, idx+1)
        mne.viz.plot_topomap(topo, pos, axes=axes, show=False)

    plt.show(block=False)

    if save_path:
        fig_.savefig(save_path + type_ + '_subplots.png', dpi=310)


def plot_average(data):
    fig_ = plt.figure()

    data_average = np.mean(data, axis=0)

    axes = fig_.add_subplot(1, 1, 1)
    mne.viz.plot_topomap(data_average, pos, axes=axes, show=False)

    plt.show(block=False)


def plot_stats_plot(data):
    frame = pd.DataFrame(data)

    pvalues = [stats.ttest_1samp(frame[idx], 0).pvalue
               for idx in range(data.shape[1])]

    fig_ = plt.figure()
    axes = fig_.add_subplot(1, 1, 1)
    mne.viz.plot_topomap(pvalues, pos, axes=axes, show=False, cmap='gist_yarg')

    plt.show(block=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pos')
    parser.add_argument('dataset', nargs='+')
    parser.add_argument('--save_path')
    cli_args = parser.parse_args()

    pos = np.loadtxt(cli_args.pos)
    save_path = cli_args.save_path

    separate_datasets = load_meditation_data(cli_args.dataset)

    # plot_in_subplots(separate_datasets['mind'], save_path=save_path)
    # plot_in_subplots(separate_datasets['anx'], save_path=save_path)
    # plot_in_subplots(separate_datasets['rest'], save_path=save_path)
    plot_in_subplots(separate_datasets['mind'] - separate_datasets['anx'], type_='mindanx',
                     save_path=save_path)
    plot_in_subplots(separate_datasets['mind'] - separate_datasets['plan'], type_='mindplan',
                     save_path=save_path)
    plot_in_subplots(separate_datasets['mind'] - separate_datasets['rest'], type_='mindrest',
                     save_path=save_path)

    import pdb; pdb.set_trace()
    print "miau"

