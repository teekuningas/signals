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


def plot_vol_stc_brainmap(brainmap, vertices, spacing, subjects_dir, axes, vmax=None, cap=0.85, cmap='RdBu_r'):


    brainmap_inc = brainmap.copy()
    brainmap_dec = brainmap.copy()

    brainmap_inc[brainmap_inc < 0] = 0
    brainmap_dec[brainmap_dec > 0] = 0
    brainmap_dec = -brainmap_dec

    if not vmax:
        vmax = np.max([np.max(brainmap_inc), np.max(brainmap_dec)])

    if np.max(brainmap_dec) > 0.01:
        factor = np.sqrt(np.max(brainmap_inc)) / np.sqrt(np.max(brainmap_dec))

        if factor > 1:
            inc_alpha = 1.0
            dec_alpha = 1.0/factor
        else:
            inc_alpha = factor
            dec_alpha = 1.0
    else:
        inc_alpha = 1.0
        dec_alpha = 0.0

    # color balancing..
    if cmap == 'RdBu_r':
        inc_alpha = 1.0*inc_alpha
        dec_alpha = 0.7*dec_alpha

    # apply vmax
    inc_alpha = inc_alpha*np.max(np.abs(brainmap_inc))/vmax
    dec_alpha = dec_alpha*np.max(np.abs(brainmap_dec))/vmax

    plot_inc = True
    plot_dec = True

    inc_bottom_cap = np.percentile(brainmap_inc, cap*100)
    if not np.allclose(brainmap_inc, 0):
        brainmap_inc[brainmap_inc <= inc_bottom_cap] = inc_bottom_cap
    else:
        plot_inc = False

    dec_bottom_cap = np.percentile(brainmap_dec, cap*100)
    if not np.allclose(brainmap_dec, 0):
        brainmap_dec[brainmap_dec <= dec_bottom_cap] = dec_bottom_cap
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

    import matplotlib as mpl
    dec_cmap = mpl.colors.ListedColormap(mpl.cm.get_cmap(cmap)(np.linspace(0.5, 0, 256)))
    inc_cmap = mpl.colors.ListedColormap(mpl.cm.get_cmap(cmap)(np.linspace(0.5, 1, 256)))

    if plot_inc:
        display.add_overlay(nifti_inc, alpha=inc_alpha, cmap=inc_cmap)

    if plot_dec:
        display.add_overlay(nifti_dec, alpha=dec_alpha, cmap=dec_cmap)

