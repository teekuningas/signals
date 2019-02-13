PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('font', size=3)
    matplotlib.use('Agg')

import pyface.qt

import sys
import csv
import argparse
import os

from collections import OrderedDict

import nibabel as nib
import mne
import numpy as np
import matplotlib.pyplot as plt 
import sklearn
import pandas as pd

from nilearn.plotting import plot_glass_brain

from sklearn.decomposition import FastICA

from scipy.signal import hilbert
from scipy.signal import decimate

# from icasso import Icasso

from signals.cibr.common import preprocess


def create_vol_stc(raw, trans, subject, noise_cov, spacing, 
                   mne_method, mne_depth, subjects_dir):
    """
    """
    
    bem = os.path.join(subjects_dir, subject, 'bem',
                       subject+'-inner_skull-bem-sol.fif')

    src_fname = os.path.join(subjects_dir, subject, 'bem',
                             subject + '-vol-' + spacing + '-src.fif')

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
        depth=mne_depth,
        fixed=False,
        limit_depth_chs=False,
        verbose='warning')

    print("Applying inverse operator..")
    stc = mne.minimum_norm.apply_inverse_raw(
        raw=raw,
        inverse_operator=inv,
        lambda2=0.1,
        method=mne_method,
        label=None,
        pick_ori=None,
        verbose='warning')

    return stc


def plot_vol_stc_brainmap_multiple(save_path, name, brainmap_inc, brainmap_dec, vertices, spacing, subjects_dir):

    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    fig_ = plt.figure()

    brainmap_inc = (brainmap_inc - np.mean(brainmap_inc)) / np.std(brainmap_inc)
    brainmap_dec = (brainmap_dec - np.mean(brainmap_dec)) / np.std(brainmap_dec)

    stc_inc = mne.source_estimate.VolSourceEstimate(
        brainmap_inc[:, np.newaxis],
        vertices,
        tstep=0.1,
        tmin=0,
        subject='fsaverage')
    stc_dec = mne.source_estimate.VolSourceEstimate(
        brainmap_dec[:, np.newaxis],
        vertices,
        tstep=0.1,
        tmin=0,
        subject='fsaverage')

    src_fname = os.path.join(subjects_dir, 'fsaverage', 'bem', 
                             'fsaverage-vol-' + spacing + '-src.fif')
    src = mne.source_space.read_source_spaces(src_fname)

    t1_fname = os.path.join(subjects_dir, 'fsaverage', 'mri', 'aseg.mgz')
    t1_img = nib.load(t1_fname)

    nifti_inc = stc_inc.as_volume(src).slicer[:, :, :, 0]
    nifti_dec = stc_dec.as_volume(src).slicer[:, :, :, 0]

    display = plot_glass_brain(t1_img, figure=fig_, display_mode='lyrz')
    display.add_overlay(nifti_inc, alpha=0.9, cmap='Reds')
    display.add_overlay(nifti_dec, alpha=0.5, cmap='Blues')
    if not save_path:
        plt.show()

    if save_path:

        brain_path = os.path.join(save_path, 'vol_brains')
        if not os.path.exists(brain_path):
            os.makedirs(brain_path)

        path = os.path.join(brain_path,
            name + '.png')

        fig_.savefig(path, dpi=620)


def plot_vol_stc_brainmap(save_path, name, brainmap, cmap, vertices, spacing, subjects_dir):

    # create necessary savepath
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    fig_ = plt.figure()

    stc = mne.source_estimate.VolSourceEstimate(
        brainmap[:, np.newaxis],
        vertices,
        tstep=0.1,
        tmin=0,
        subject='fsaverage')

    src_fname = os.path.join(subjects_dir, 'fsaverage', 'bem', 
                             'fsaverage-vol-' + spacing + '-src.fif')
    src = mne.source_space.read_source_spaces(src_fname)

    t1_fname = os.path.join(subjects_dir, 'fsaverage', 'mri', 'aseg.mgz')
    t1_img = nib.load(t1_fname)

    nifti = stc.as_volume(src).slicer[:, :, :, 0]

    display = plot_glass_brain(t1_img, figure=fig_, display_mode='lyrz')
    # display.add_overlay(nifti, alpha=0.75)
    display.add_overlay(nifti, alpha=0.75, cmap=cmap)

    if not save_path:
        plt.show()

    if save_path:

        brain_path = os.path.join(save_path, 'vol_brains')
        if not os.path.exists(brain_path):
            os.makedirs(brain_path)

        path = os.path.join(brain_path,
            name + '.png')

        fig_.savefig(path, dpi=620)


def extract_intervals_meditation(events, sfreq, first_samp, tasks):
    """ meditation intervals """
    intervals = OrderedDict()

    trigger_info = [
        ('mind', 10),
        ('rest', 11),
        ('plan', 12),
        ('anx', 13),
    ]

    for name, event_id in trigger_info:
        intervals[name] = []

    for idx, event in enumerate(events):
        for name, event_id in trigger_info:
            if name not in tasks:
                continue
            if event[2] == event_id:
                ival_start = event[0] + 2*sfreq
                try:
                    ival_end = events[idx+1][0] - 2*sfreq
                except:
                    # last trigger (rest)
                    ival_end = event[0] + 120*sfreq - 2*sfreq

                intervals[name].append((
                    (ival_start - first_samp) / sfreq,
                    (ival_end - first_samp) / sfreq))

    return intervals


def gather_background_data_meditation(subject, raw, events, csv_path, tasks):

    data = []
    header = []

    # first create state data from exp sampling

    subject_code = subject.split('subject_')[-1]

    sfreq = raw.info['sfreq']
    first_samp = raw.first_samp

    trigger_info = [
        ('mind', 10), 
        ('rest', 11),
        ('plan', 12),
        ('anx', 13),
    ]

    ivals = []
    ival_names = []
    for idx, event in enumerate(events):
        for name, event_id in trigger_info:
            # state data only for nonrest tasks
            if name == 'rest':
                continue
            if event[2] == event_id:
                # ival_start = event[0] + 2*sfreq
                ival_start = event[0]
                try:
                    # ival_end = events[idx+1][0] - 2*sfreq
                    ival_end = events[idx+1][0]
                except:
                    # last trigger (rest)
                    # ival_end = event[0] + 120*sfreq - 2*sfreq
                    ival_end = event[0] + 120*sfreq

                ivals.append((ival_start - first_samp,
                              ival_end - first_samp))
                ival_names.append(name)

    def create_state_data(fname):
        data_row = np.array([None]*len(raw.times))
        with open(os.path.join(csv_path, fname), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', skipinitialspace=True)
            for csv_row in reader:
                if csv_row[0].zfill(3) != subject_code:
                    continue

                if len(ivals)/2 != len(csv_row[1:]):
                    print(str(len(ivals)) + str(len(csv_row)))
                    import pdb; pdb.set_trace()
                    raise Exception('Amount of tasks does not match in csv and raw')

                task_values = {}
                for ival_idx, ival in enumerate(ivals):
                    ival_name = ival_names[ival_idx]
                    if ival_name not in task_values:
                        task_values[ival_name] = []
                    val = float(csv_row[1:][ival_idx/2])
                    task_values[ival_name].append(val)

                # values = [float(val) for val in csv_row[1:]]
                # mean_, std_ = np.mean(values), np.std(values)
                # import pdb; pdb.set_trace()

                for ival_idx, ival in enumerate(ivals):
                    if ival_names[ival_idx] not in tasks:
                        continue
                    mean_ = np.mean(task_values[ival_names[ival_idx]])
                    std_ = np.std(task_values[ival_names[ival_idx]])
                    val = (float(csv_row[1:][ival_idx/2]) - mean_) / std_
                    data_row[int(ival[0]):int(ival[1])] = val

        # interpolate Nones
        data_row[data_row == None] = np.nan
        data_row = pd.DataFrame(data_row.astype(np.float)).interpolate(
            limit_direction='both').values[:, 0]

        return data_row

    data.append(create_state_data('feeling.csv'))
    data.append(create_state_data('focused.csv'))
    header.append('feeling')
    header.append('focused')

    data_row = np.array([None]*len(raw.times))

    for ival_idx, ival in enumerate(ivals):
        if ival_names[ival_idx] == tasks[0]:
            val = 0.0
        elif ival_names[ival_idx] == tasks[1]:
            val = 1.0
        else:
            continue
        data_row[int(ival[0]):int(ival[1])] = val

    # interpolate Nones
    data_row[data_row == None] = np.nan
    data_row = pd.DataFrame(data_row.astype(np.float)).interpolate(
        limit_direction='both').values[:, 0]
    # data_row[data_row == None] = 0.5

    data.append(data_row)
    header.append('Task')

    # then add trait data
    with open(os.path.join(csv_path, 'tests.csv'), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', skipinitialspace=True)
        csv_header = next(reader)
        for csv_row in reader:
            if csv_row[0].zfill(3) != subject_code:
                continue

            for item_idx, item in enumerate(csv_row[1:]):
                allowed = []
                # allowed = ['BDI', 'BAI', 'BIS', 'BasDrive', 'BasRR', 'BasFS',
                #            'MedLength', 'MedFreq', 'MedExp']
                # allowed = ['BDI', 'BAI', 'BIS', 'MedExp']
                if csv_header[1:][item_idx] not in allowed:
                    continue
                data_row = np.array([float(item)]*len(raw.times))
                data.append(data_row)
                header.append(csv_header[1:][item_idx])

    # data.append(data[2]*data[-1])
    # header.append('Task*MedExp')
    # data.append(data[2]*data[-2])
    # header.append('Task*BIS')
    # data.append(data[2]*data[-3])
    # header.append('Task*BAI')
    # data.append(data[2]*data[-4])
    # header.append('Task*BDI')

    return header, np.array(data)


def get_amount_of_components(data, explained_var):

    from sklearn.decomposition import PCA
    pca = PCA(whiten=True, copy=True)

    data = pca.fit_transform(data.T)

    n_components = np.sum(pca.explained_variance_ratio_.cumsum() <=
                           explained_var)
    return n_components


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contrast')
    parser.add_argument('--raw')
    parser.add_argument('--empty', nargs='+')
    parser.add_argument('--save-path')
    parser.add_argument('--csv-path')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    src_method = 'vol'
    mne_method, mne_depth = 'dSPM', None
    band = (16, 28)
    sampling_rate_raw = 100.0
    tasks = ['mind', cli_args.contrast]

    surf_spacing = 'ico3'
    vol_spacing = '10'
    page = 4

    # hilbert ica settings
    sampling_rate_hilbert = 1.0

    # process similarly to input data 
    empty_paths = cli_args.empty
    empty_raws = []
    for fname in empty_paths:
        raw = mne.io.Raw(fname, preload=True)
        raw.resample(sampling_rate_raw)
        raw, _ = preprocess(raw, filter_=band)
        empty_raws.append(raw)
    empty_raw = mne.concatenate_raws(empty_raws)
    empty_raws = []

    print("Creating sensor covariance matrix..")
    noise_cov = mne.compute_raw_covariance(
        empty_raw, 
        method='empirical')

    current_time = 0
    subject_data = {}

    path = cli_args.raw

    folder = os.path.dirname(path)
    fname = os.path.basename(path)

    subject = '_'.join(fname.split('_')[:2])

    print("Using MRI subject: " + subject)

    trans = os.path.join(folder, subject + '-trans.fif')

    print("Handling " + path)

    raw = mne.io.Raw(path, preload=True)
    raw.resample(sampling_rate_raw)
    raw, events = preprocess(raw, filter_=band, min_duration=1)

    intervals = extract_intervals_meditation(
        events, 
        raw.info['sfreq'], 
        raw.first_samp,
        tasks)

    # print "Gathering background data"
    # bg_header, bg_data = gather_background_data_meditation(
    #     subject, raw, events, cli_args.csv_path,
    #     tasks)

    stc = create_vol_stc(
        raw=raw, 
        trans=trans, 
        subject=subject, 
        noise_cov=noise_cov, 
        spacing=vol_spacing,
        mne_method=mne_method,
        mne_depth=mne_depth,
        subjects_dir=subjects_dir) 

    subject_data['name'] = subject
    subject_data['intervals'] = intervals
    # subject_data['bg_header'] = bg_header
    subject_data['start'] = current_time

    def prepare_hilbert(data):
        # get envelope as abs of analytic signal
        datas = np.array_split(data, 20, axis=0)
        envs = [np.abs(hilbert(arr)) for arr in datas]
        env = np.concatenate(envs, axis=0)
        # decimate first with five
        decimated = decimate(env, 5)
        # and then the rest
        factor = int(((sampling_rate_raw / 5.0) / sampling_rate_hilbert))
        decimated = decimate(decimated, factor)
        return decimated

    # decim_factor = int(((sampling_rate_raw / 5.0) / sampling_rate_hilbert))
    # bg_data = decimate(decimate(bg_data, 5), decim_factor)

    print("Prepare using hilbert")
    data = prepare_hilbert(stc.data)
    print("Hilbert done")

    subject_data['data'] = data
    # subject_data['bg_data'] = bg_data

    current_time += len(raw.times) / raw.info['sfreq']

    vertices = stc.vertices

    explained_var = 0.95
    n_explained_components = get_amount_of_components(subject_data['data'], explained_var)
    n_components = max(min(n_explained_components, 30), 7)
    # n_components = n_explained_components

    print("Explained var " + str(explained_var) + " yields " + str(n_explained_components) + 
          " components.")
    print("Using " + str(n_components))

    ica_params = {
        'n_components': n_components,
        'algorithm': 'parallel',
        'whiten': True,
        'max_iter': 100000,
        'tol': 0.0000000001
    }
    ica = FastICA(**ica_params)
    ica.fit(subject_data['data'].T)
    ica_unmixing = ica.components_
    ica_mixing = np.linalg.pinv(ica_unmixing)

    independent_data = np.dot(ica_unmixing, subject_data['data'])

    savename = subject_data['name'] + '_' + tasks[0] + '_' + tasks[1]

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    ###

    def classify_plot_and_save(blocktype, interval_dict):

        # X = [] 
        # y = []
        # length = 4
        # for key, ivals in interval_dict.items():
        #     if key not in tasks:
        #         continue
        #     for ival in ivals:
        #         subivals = [(istart, istart + length) for istart in range(int(ival[0]), int(ival[1]), length)]
        #         for subival in subivals:
        #             start = int(subival[0]*sampling_rate_hilbert)
        #             end = int(subival[1]*sampling_rate_hilbert)
        #             X.append(np.mean(independent_data[:, start:end], axis=1))
        #             if key == tasks[0]:
        #                 y.append(0)
        #             if key == tasks[1]:
        #                 y.append(1)

        # parameters = {
        #     'alpha': [1e-6, 1e-5, 1e-4, 1e-3],
        #     'l1_ratio': [0.6, 0.7, 0.8, 0.9, 1.0],
        # }
        # alg = sklearn.linear_model.SGDClassifier(
        #     loss='log',
        #     penalty='elasticnet',
        #     max_iter=10000,
        #     tol=None,
        #     class_weight='balanced',
        # )
        # clf = sklearn.model_selection.GridSearchCV(
        #     estimator=alg,
        #     param_grid=parameters,
        #     cv=4)
        # clf.fit(X, y)

        # scores = sklearn.model_selection.cross_val_score(clf.best_estimator_, X, y, cv=4)
        # l1_ratio = clf.best_estimator_.l1_ratio
        # clf_alpha = clf.best_estimator_.alpha

        # print("Stats for :" + str(blocktype))
        # print("Results: " + str(scores))
        # print("L1 ratio: " + str(l1_ratio))
        # print("Alpha: " + str(clf_alpha))

        # if save_path:
        #     info_path = os.path.join(save_path, 'subject_info')
        #     if not os.path.exists(info_path):
        #         os.makedirs(info_path)

        #     path = os.path.join(info_path,
        #         savename + '_' + blocktype + '.csv')

        #     with open(path, 'w') as f:
        #         f.write(', '.join([
        #             savename,
        #             str(np.mean(scores)),
        #             str(l1_ratio),
        #             str(clf_alpha)]))

        X = [] 
        y = []
        length = 4
        for key, ivals in interval_dict.items():
            if key not in tasks:
                continue
            for ival in ivals:
                subivals = [(istart, istart + length) for istart in 
                            range(int(ival[0]), int(ival[1]), length)]
                for subival in subivals:
                    start = int(subival[0]*sampling_rate_hilbert)
                    end = int(subival[1]*sampling_rate_hilbert)
                    X.append(np.mean(subject_data['data'][:, start:end], axis=1))
                    if key == tasks[0]:
                        y.append(0)
                    if key == tasks[1]:
                        y.append(1)

        X_ = np.array(X)
        y_ = np.array(y)

        difference_brainmap = np.mean(X_[y_==1], axis=0) - np.mean(X_[y_==0], axis=0)
        increase_map = difference_brainmap.copy()
        increase_map[increase_map<0] = 0
        decrease_map = difference_brainmap.copy()
        decrease_map[decrease_map>0] = 0
        decrease_map = -decrease_map

        plot_vol_stc_brainmap(
            save_path, savename + '_increase_map_noica_' + blocktype, 
            increase_map, 'Reds', vertices, vol_spacing, subjects_dir) 
        plot_vol_stc_brainmap(
            save_path, savename + '_decrease_map_noica_' + blocktype, 
            decrease_map, 'Blues', vertices, vol_spacing, subjects_dir) 

        plot_vol_stc_brainmap_multiple(
            save_path, savename + '_combined_map_noica_' + blocktype, 
            increase_map, decrease_map, vertices, vol_spacing, subjects_dir) 

        if save_path:
            data_path = os.path.join(save_path, 'data')
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            path = os.path.join(data_path, savename + '_noica_' + blocktype + '.csv')

            with open(path, 'w') as f:
                f.write(', '.join([str(elem) for elem in vertices.tolist()]) + '\n')
                f.write(', '.join([str(elem) for elem in difference_brainmap.tolist()]))

    intervals = subject_data['intervals']
    block1_task0 = intervals[tasks[0]][:int(len(intervals[tasks[0]])/2)]
    block1_task1 = intervals[tasks[1]][:int(len(intervals[tasks[1]])/2)]
    block2_task0 = intervals[tasks[0]][int(len(intervals[tasks[0]])/2):]
    block2_task1 = intervals[tasks[1]][int(len(intervals[tasks[1]])/2):]

    block1_intervals = OrderedDict()
    block1_intervals[tasks[0]] = block1_task0
    block1_intervals[tasks[1]] = block1_task1
    block2_intervals = OrderedDict()
    block2_intervals[tasks[0]] = block2_task0
    block2_intervals[tasks[1]] = block2_task1

    # get accuracy by training with session 1 
    # and testing with session 2
    block1_X = [] 
    block1_y = []
    length = 4
    for key, ivals in block1_intervals.items():
        if key not in tasks:
            continue
        for ival in ivals:
            subivals = [(istart, istart + length) for istart in range(int(ival[0]), int(ival[1]), length)]
            for subival in subivals:
                start = int(subival[0]*sampling_rate_hilbert)
                end = int(subival[1]*sampling_rate_hilbert)
                block1_X.append(np.mean(independent_data[:, start:end], axis=1))
                if key == tasks[0]:
                    block1_y.append(0)
                if key == tasks[1]:
                    block1_y.append(1)
    block2_X = [] 
    block2_y = []
    length = 4
    for key, ivals in block2_intervals.items():
        if key not in tasks:
            continue
        for ival in ivals:
            subivals = [(istart, istart + length) for istart in range(int(ival[0]), int(ival[1]), length)]
            for subival in subivals:
                start = int(subival[0]*sampling_rate_hilbert)
                end = int(subival[1]*sampling_rate_hilbert)
                block2_X.append(np.mean(independent_data[:, start:end], axis=1))
                if key == tasks[0]:
                    block2_y.append(0)
                if key == tasks[1]:
                    block2_y.append(1)

    parameters = {
        'alpha': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        'l1_ratio': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    alg = sklearn.linear_model.SGDClassifier(
        loss='log',
        penalty='elasticnet',
        max_iter=10000,
        tol=None,
        class_weight='balanced',
    )
    clf = sklearn.model_selection.GridSearchCV(
        estimator=alg,
        param_grid=parameters,
        cv=4)

    clf.fit(block1_X, block1_y)

    scores = sklearn.model_selection.cross_val_score(clf.best_estimator_, block1_X, block1_y, cv=4)
    train_score = np.mean(scores)
    l1_ratio = clf.best_estimator_.l1_ratio
    clf_alpha = clf.best_estimator_.alpha
    test_score = clf.best_estimator_.score(block2_X, block2_y)

    print("Main classification:")
    print("Results: " + str(scores))
    print("Train CV score: " + str(train_score))
    print("L1 ratio: " + str(l1_ratio))
    print("Alpha: " + str(clf_alpha))
    print("Validation score: " + str(test_score))

    if save_path:
        info_path = os.path.join(save_path, 'subject_info')
        if not os.path.exists(info_path):
            os.makedirs(info_path)

        path = os.path.join(info_path,
            savename + '.csv')

        with open(path, 'w') as f:
            f.write(', '.join([
                savename,
                str(train_score),
                str(test_score),
                str(l1_ratio),
                str(clf_alpha)]))

    classify_plot_and_save('both', intervals)
    classify_plot_and_save('block1', block1_intervals)
    classify_plot_and_save('block2', block2_intervals)
    

