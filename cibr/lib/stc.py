import os

import mne
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt 

from nilearn.plotting import plot_glass_brain


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


def plot_vol_stc_brainmap_old(save_path, name, brainmap, vertices, spacing, subjects_dir, axes=None, verbose=False, folder_name='vol_brains', cap=0.85):


    brainmap_inc = brainmap.copy()
    brainmap_dec = brainmap.copy()

    brainmap_inc[brainmap_inc < 0] = 0
    brainmap_dec[brainmap_dec > 0] = 0
    brainmap_dec = -brainmap_dec

    plot_inc = True
    plot_dec = True

    if verbose:
        print("Inc max before scaling: " + str(np.max(brainmap_inc)))
        print("Dec max before scaling: " + str(np.max(brainmap_dec)))

    vmax = np.max([np.max(brainmap_inc), np.max(brainmap_dec)])

    brainmap_inc = brainmap_inc / vmax
    brainmap_dec = brainmap_dec / vmax

    if np.max(brainmap_dec) > 0.01:
        factor = np.sqrt(np.max(brainmap_inc)) / np.sqrt(np.max(brainmap_dec))

        if verbose:
            print("Factor is: " + str(factor))

        if factor > 1:
            red_alpha = 1.0
            blue_alpha = 1.0/factor
        else:
            blue_alpha = 1.0
            red_alpha = factor
    else:
        if verbose:
            print("Selecting default values for alphas.")
        red_alpha = 1.0
        blue_alpha = 0.0

    # color balancing..
    red_alpha = 1.0*red_alpha
    blue_alpha = 0.7*blue_alpha

    if verbose:
        print('Red alpha: ' + str(red_alpha))
        print('Blue alpha: ' + str(blue_alpha))

    inc_bottom_cap = np.percentile(brainmap_inc, cap*100)
    if not np.allclose(brainmap_inc, 0):
        brainmap_inc[brainmap_inc < inc_bottom_cap] = inc_bottom_cap
    else:
        plot_inc = False

    dec_bottom_cap = np.percentile(brainmap_dec, cap*100)
    if not np.allclose(brainmap_dec, 0):
        brainmap_dec[brainmap_dec < dec_bottom_cap] = dec_bottom_cap
    else:
        plot_dec = False

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
    src = mne.source_space.read_source_spaces(src_fname, verbose='error')

    aseg_fname = os.path.join(subjects_dir, 'fsaverage', 'mri', 'aseg.mgz')
    aseg_img = nib.load(aseg_fname)

    nifti_inc = stc_inc.as_volume(src).slicer[:, :, :, 0]
    nifti_dec = stc_dec.as_volume(src).slicer[:, :, :, 0]

    if axes == None:
        fig_, axes = plt.subplots()

    display = plot_glass_brain(aseg_img, axes=axes, display_mode='lzr')

    if plot_inc:
        display.add_overlay(nifti_inc, alpha=red_alpha, cmap='Reds')

    if plot_dec:
        display.add_overlay(nifti_dec, alpha=blue_alpha, cmap='Blues')

    if save_path:

        brain_path = os.path.join(save_path, folder_name)
        if not os.path.exists(brain_path):
            os.makedirs(brain_path)

        path = os.path.join(brain_path,
            name + '.png')

        fig_.savefig(path, dpi=310)


def plot_vol_stc_brainmap(brainmap, vertices, spacing, subjects_dir, axes, vmax=None, cap=0.85):


    brainmap_inc = brainmap.copy()
    brainmap_dec = brainmap.copy()

    brainmap_inc[brainmap_inc < 0] = 0
    brainmap_dec[brainmap_dec > 0] = 0
    brainmap_dec = -brainmap_dec

    if not vmax:
        vmax = np.max([np.max(brainmap_inc), np.max(brainmap_dec)])

    brainmap_inc = brainmap_inc / vmax
    brainmap_dec = brainmap_dec / vmax

    if np.max(brainmap_dec) > 0.01:
        factor = np.sqrt(np.max(brainmap_inc)) / np.sqrt(np.max(brainmap_dec))

        if factor > 1:
            red_alpha = 1.0
            blue_alpha = 1.0/factor
        else:
            blue_alpha = 1.0
            red_alpha = factor
    else:
        red_alpha = 1.0
        blue_alpha = 0.0

    # color balancing..
    red_alpha = 1.0*red_alpha
    blue_alpha = 0.7*blue_alpha

    # apply vmax
    red_alpha = red_alpha*np.max(np.abs(brainmap_inc))/vmax
    blue_alpha = blue_alpha*np.max(np.abs(brainmap_dec))/vmax

    plot_inc = True
    plot_dec = True

    inc_bottom_cap = np.percentile(brainmap_inc, cap*100)
    if not np.allclose(brainmap_inc, 0):
        brainmap_inc[brainmap_inc < inc_bottom_cap] = inc_bottom_cap
    else:
        plot_inc = False

    dec_bottom_cap = np.percentile(brainmap_dec, cap*100)
    if not np.allclose(brainmap_dec, 0):
        brainmap_dec[brainmap_dec < dec_bottom_cap] = dec_bottom_cap
    else:
        plot_dec = False

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
    src = mne.source_space.read_source_spaces(src_fname, verbose='error')

    aseg_fname = os.path.join(subjects_dir, 'fsaverage', 'mri', 'aseg.mgz')
    aseg_img = nib.load(aseg_fname)

    nifti_inc = stc_inc.as_volume(src).slicer[:, :, :, 0]
    nifti_dec = stc_dec.as_volume(src).slicer[:, :, :, 0]

    display = plot_glass_brain(aseg_img, axes=axes, display_mode='lzr')

    if plot_inc:
        display.add_overlay(nifti_inc, alpha=red_alpha, cmap='Reds')

    if plot_dec:
        display.add_overlay(nifti_dec, alpha=blue_alpha, cmap='Blues')

