PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('font', size=6)
    matplotlib.use('Agg')

import os
# os.environ['ETS_TOOLKIT'] = 'qt4'
# os.environ['QT_API'] = 'pyqt4'
# os.environ['ETS_TOOLKIT'] = 'wx'

import sys
import argparse

# import pyface.qt

import mne
import numpy as np
import matplotlib.pyplot as plt 

import scipy.signal

from signals.cibr.ica.complex_ica import complex_ica

from signals.cibr.common import load_raw
from signals.cibr.common import preprocess
from signals.cibr.common import get_correlations
from signals.cibr.common import calculate_stft
from signals.cibr.common import arrange_as_matrix
from signals.cibr.common import arrange_as_tensor
from signals.cibr.common import plot_topomaps
from signals.cibr.common import plot_mean_spectra
from signals.cibr.common import plot_subject_spectra
from signals.cibr.common import get_rest_intervals


def visualize_components_in_cortex(inv, maps, raw_info, save_path):

    loop_idx = 0
    while True:
        if save_path:
            component_idx = loop_idx
        else:

            input_ = raw_input("Choose component to plot: ")
            try:
                component_idx = int(input_) - 1
            except:
                break

        try:
            evoked = mne.EvokedArray(maps[:, component_idx, np.newaxis],
                raw_info)
        except:
            break

        stc = mne.minimum_norm.apply_inverse(
                evoked=evoked, 
                inverse_operator=inv,
                lambda2=0.1,
                method='MNE',
                pick_ori=None,
                verbose='warning')

        fmin = np.percentile(stc.data, 85)
        fmid = np.percentile(stc.data, 92)
        fmax = np.percentile(stc.data, 97)

        brain = stc.plot(hemi='split', views=['med', 'lat'], smoothing_steps=150,
                         surface='inflated',
                         clim={'kind': 'value', 'lims': [fmin, fmid, fmax]})

        if save_path:

            brain_path = os.path.join(save_path, 'brains')
            if not os.path.exists(brain_path):
                os.makedirs(brain_path)

            path = os.path.join(brain_path, 
                'comp_' + str(component_idx+1).zfill(2) + '.png')

            brain.save_image(path)

        loop_idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raws', nargs='+')
    parser.add_argument('--empty', nargs='+')
    parser.add_argument('--save-path')
    cli_args = parser.parse_args()

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    raws = []
    names = []
    splits_in_samples = [0]
    for path_idx, path in enumerate(cli_args.raws):

        print(path)

        raw = load_raw(path)
        raw, events = preprocess(raw)

        raw.resample(50)

        raws.append(raw)
        names.append(raw.filenames[0].split('/')[-1].split('.fif')[0])

        splits_in_samples.append(splits_in_samples[-1] + len(raw))

    raw = mne.concatenate_raws(raws)

    sfreq = raw.info['sfreq']
    n_components = 20
    page = 5
    conveps = 1e-8
    maxiter = 30000
    hpass = 4
    lpass = 16

    # window_in_seconds = 2
    # window_in_samples = np.power(2, np.ceil(np.log(
    #     sfreq * window_in_seconds)/np.log(2)))
    # overlap_in_samples = (window_in_samples * 3) / 4
    window_in_samples = int(raw.info['sfreq']*2)*5
    overlap_in_samples = int(window_in_samples/2.0)

    # eo_ivals, ec_ivals, total_ivals = get_rest_intervals(splits_in_samples, sfreq)

    # create new data based only on ec
    # raws = []
    # index_count = 0
    # total_ivals = []
    # for ival in ec_ivals:
    #     total_ival = (index_count, index_count + ival[1] + ival[0])
    #     index_count += ival[1] - ival[0]
    #     total_ivals.append(total_ival)
    #     raws.append(mne.io.RawArray(raw._data[:, ival[0]:ival[1]], raw.info))

    # raw = mne.concatenate_raws(raws)
    # print total_ivals

    # set seed
    np.random.seed(10)

    freqs, times, data, _ = (
        calculate_stft(raw._data, sfreq, window_in_samples, 
                       overlap_in_samples, hpass, lpass))

    shape, data = data.shape, arrange_as_matrix(data)

    data, mixing, dewhitening, _, _, mean = complex_ica(
        data, n_components, conveps=conveps, maxiter=maxiter)

    data = arrange_as_tensor(data, shape)
    
    # seprate to subjects
    spectrums = []
    for idx in range(len(raws)):
        start = splits_in_samples[idx]*data.shape[2]/float(len(raw))
        if idx != len(raws) - 1:
            end = splits_in_samples[idx+1]*data.shape[2]/float(len(raw))
        else:
            end = data.shape[2]
        print('Subject ' + str(idx+1) + ': ' + str(start) + ' - ' + 
              str(end) + ' (' + str(data.shape[2]) + ')')
        spectrums.append(np.mean(np.abs(data[:, :, start:end]), axis=2))
    spectrums = np.array(spectrums)

    header = [''] + [str(freq) for freq in freqs]
    lines = []
    for subject_idx in range(spectrums.shape[0]):
        for component_idx in range(spectrums.shape[1]):
            name = (cli_args.raws[subject_idx].split('/')[-1].split('.fif')[0] + 
                    ' (' + str(component_idx+1) + ')')
            line = [name] + [str(val) for val in spectrums[subject_idx, component_idx]]
            lines.append(line)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, 'data.csv'), 'wb') as f:
            f.write(';'.join(header) + '\n')
            for line in lines:
                f.write(';'.join(line) + '\n')

    print("Plotting brainmaps.")
    plot_topomaps(save_path, dewhitening, mixing, mean, raw.info,
                   page, range(dewhitening.shape[1]))

    subject = 'FDMSA_D12_9'
    subject_code = subject.split('FDMSA_')[-1]
    subject_raw_path = '/nashome1/erpipehe/data/fdmsa/rest/processed/FDMSA_restpre_' + subject_code + '_tsss_mc.fif'
    subject_raw = mne.io.Raw(subject_raw_path, preload=True)
    subject_raw, _ = preprocess(subject_raw)
    subject_raw = subject_raw.resample(raw.info['sfreq'])
 
    trans = '/nashome1/erpipehe/data/fdmsa/rest/processed/' + subject + '-trans.fif'
    subjects_dir = '/nashome1/erpipehe/data/fdmsa/rest/RECON'

    os.environ['SUBJECTS_DIR'] = subjects_dir
    os.environ['SUBJECT'] = subject

    bem = os.path.join(subjects_dir, subject, 'bem',
                       subject+'-inner_skull-bem-sol.fif')

    print("Computing source space..")
    src = mne.setup_source_space(
        subject=subject,
        spacing='ico4',
        surface='white',
        subjects_dir=subjects_dir,
        verbose='warning')

    print("Creating forward solution..")
    fwd = mne.make_forward_solution(
        info=subject_raw.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        verbose='warning')

    print("Gathering noise covariance..")
    empty_raws = []
    for path in cli_args.empty:
        tmp_raw = mne.io.Raw(path, preload=True)
        tmp_raw, _ = preprocess(tmp_raw)
        empty_raws.append(tmp_raw)
    empty_raw = mne.concatenate_raws(empty_raws)
    empty_raw = empty_raw.resample(raw.info['sfreq'])
    noise_cov = mne.compute_raw_covariance(empty_raw)

    print("Making inverse operator..")
    inv = mne.minimum_norm.make_inverse_operator(
        info=subject_raw.info, 
        forward=fwd, 
        noise_cov=noise_cov,
        loose=1.0,
        depth=None,
        fixed=False,
        limit_depth_chs=True,
        verbose='warning')

    maps = np.abs(np.matmul(dewhitening, mixing) + mean[:, np.newaxis])

    print("Plotting brains..")
    visualize_components_in_cortex(inv, maps, subject_raw.info, save_path)


