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

    # rescale between -1 and 1 to make plotting nonbuggy
    max_val = np.max(np.abs(brainmap))
    brainmap = brainmap / max_val

    brainmap_inc = brainmap.copy()
    brainmap_dec = brainmap.copy()

    brainmap_inc[brainmap_inc < 0] = 0
    brainmap_dec[brainmap_dec > 0] = 0
    brainmap_dec = -brainmap_dec

    if not vmax:
        vmax = np.max([np.max(brainmap_inc), np.max(brainmap_dec)])
    else:
        vmax = vmax / max_val

    if np.max(np.abs(brainmap_dec)) / np.max(np.abs(brainmap_inc)) > 0.01:
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
    mu, sigma = inc_bottom_cap, np.max(brainmap_inc) / 10000
    for idx in range(len(brainmap_inc)):
        if brainmap_inc[idx] <= mu:
            brainmap_inc[idx] = sigma * np.random.randn() + mu
            if brainmap_inc[idx] < 0:
                brainmap_inc[idx] = -brainmap_inc[idx]

    # brainmap_inc[brainmap_inc <= inc_bottom_cap] = inc_bottom_cap
    # if not np.allclose(brainmap_inc, 0):
    #     brainmap_inc[brainmap_inc <= inc_bottom_cap] = inc_bottom_cap
    # else:
    #     plot_inc = False

    dec_bottom_cap = np.percentile(brainmap_dec, cap*100)
    mu, sigma = dec_bottom_cap, np.max(brainmap_dec) / 10000
    for idx in range(len(brainmap_dec)):
        if brainmap_dec[idx] <= mu:
            brainmap_dec[idx] = sigma * np.random.randn() + mu
            if brainmap_dec[idx] < 0:
                brainmap_dec[idx] = -brainmap_dec[idx]

    # brainmap_dec[brainmap_dec <= dec_bottom_cap] = dec_bottom_cap
    # if not np.allclose(brainmap_dec, 0):
    #     brainmap_dec[brainmap_dec <= dec_bottom_cap] = dec_bottom_cap
    # else:
    #     plot_dec = False

    stc_inc = mne.source_estimate.VolSourceEstimate(
        brainmap_inc[:, np.newaxis],
        [vertices],
        tstep=0.1,
        tmin=0,
        subject='fsaverage')
    stc_dec = mne.source_estimate.VolSourceEstimate(
        brainmap_dec[:, np.newaxis],
        [vertices],
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

    display = plot_glass_brain(aseg_img, axes=axes, display_mode='lzr',
                               annotate=False)

    import matplotlib as mpl
    dec_cmap = mpl.colors.ListedColormap(mpl.cm.get_cmap(cmap)(np.linspace(0.5, 0, 256)))
    inc_cmap = mpl.colors.ListedColormap(mpl.cm.get_cmap(cmap)(np.linspace(0.5, 1, 256)))

    if plot_inc:
        display.add_overlay(nifti_inc, alpha=inc_alpha, cmap=inc_cmap,
                            interpolation='none', resampling_interpolation=None)

    if plot_dec:
        display.add_overlay(nifti_dec, alpha=dec_alpha, cmap=dec_cmap,
                            interpolation='none', resampling_interpolation=None)

def plot_vol_stc_labels(brainmap, vertices, spacing, subjects_dir, ax, n_labels=10):
    labels, label_data, label_vertices = get_vol_labeled_data(
        np.array(brainmap), vertices, subjects_dir, spacing)

    sorted_idxs = np.argsort(-np.array(label_data))
    sorted_labels = [labels[idx] for idx in sorted_idxs]
    sorted_label_data = [label_data[idx] for idx in sorted_idxs]
    sorted_label_vertices = [label_vertices[idx] for idx in sorted_idxs]

    labels = sorted_labels[:n_labels]
    label_data = sorted_label_data[:n_labels]
    label_vertices = sorted_label_vertices[:n_labels]

    aseg_fname = os.path.join(subjects_dir, 'fsaverage', 'mri', 'aseg.mgz')
    aseg_img = nib.load(aseg_fname)

    display = plot_glass_brain(aseg_img, axes=ax, display_mode='lzr',
                               annotate=False)

    # all_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
    #               'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 
    #               'tab:olive', 'tab:cyan']

    from random import randint
    all_colors = []
    for idx in range(n_labels):
        all_colors.append('#%06X' % randint(0, 0xFFFFFF))

    lws = np.copy(np.abs(label_data))
    lws = lws - np.min(lws)
    lws = lws / np.max(lws)

    from matplotlib.lines import Line2D

    custom_lines = []
    for label_idx in range(len(labels)):
        custom_lines.append(Line2D([0], [0], color=all_colors[label_idx], lw=3+3*lws[label_idx]))

    ax.legend(custom_lines, labels)

    for label_idx in range(len(labels)):
        mask = np.zeros(vertices.shape)

        for vertex in label_vertices[label_idx]:
            mask[np.where(vertices == vertex)[0][0]] = 1

        stc = mne.source_estimate.VolSourceEstimate(
            mask[:, np.newaxis],
            [vertices],
            tstep=0.1,
            tmin=0,
            subject='fsaverage')

        src_fname = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                 'fsaverage-vol-' + spacing + '-src.fif')
        src = mne.source_space.read_source_spaces(src_fname, verbose='error')

        nifti = stc.as_volume(src).slicer[:, :, :, 0]

        import matplotlib as mpl
        cmaplist = [mpl.colors.to_rgba(all_colors[label_idx]), (1.0, 1.0, 1.0, 0.0)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, N=10)

        display.add_overlay(nifti, cmap=cmap,
                            interpolation='none', resampling_interpolation=None)


def get_vol_labeled_data(brainmap, vertices, subjects_dir, spacing='10'):

    aseg_path = os.path.join(subjects_dir, 'fsaverage', 'mri', 'aparc+aseg.mgz')
    labels = mne.get_volume_labels_from_aseg(aseg_path)

    stc = mne.source_estimate.VolSourceEstimate(
        brainmap[:, np.newaxis],
        [vertices],
        tstep=0.1,
        tmin=0,
        subject='fsaverage')

    src_fname = os.path.join(subjects_dir, 'fsaverage', 'bem',
                             'fsaverage-vol-' + spacing + '-src.fif')
    src = mne.source_space.read_source_spaces(src_fname, verbose='error')

    labeled_data = []
    nonzero_labels = []
    vertices = []
    for label in labels:
        if not label.startswith('ctx'):
            continue
        restricted = stc.in_label(label, aseg_path, src)
        if restricted.data.shape[0] == 0:
            continue

        nonzero_labels.append(label)
        vertices.append(restricted.vertices[0])
        labeled_data.append(np.max(np.abs(restricted.data)))

    return np.array(nonzero_labels), np.array(labeled_data), vertices
