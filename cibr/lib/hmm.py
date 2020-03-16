from itertools import groupby

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches


def plot_state_series(state_chain_by_subject, tmin, tmax, sfreq, names, 
                      state_colors=None, task_annotations=None, show=False,
                      probabilistic=True, decimate_to=1.0):
    """
    """
    if decimate_to:
        window = decimate_to*sfreq  # decimate_to as s
        decim_st_ch_by_subj = []
        for subj_chain in state_chain_by_subject:
            decimated_subj_chain = []
            for tstart in np.arange(0, subj_chain.shape[0], window):
                decimated_subj_chain.append(np.mean(subj_chain[int(tstart):int(tstart+window)], axis=0)[np.newaxis, :])
            decim_st_ch_by_subj.append(np.concatenate(decimated_subj_chain, axis=0))
        state_chain_by_subject = decim_st_ch_by_subj
        sfreq = 1/decimate_to

    n_states = state_chain_by_subject[0].shape[1]

    # generate colors if not given
    if not state_colors:
        cmap = plt.cm.get_cmap('gist_rainbow', n_states)
        state_colors = [cmap(idx) for idx in range(n_states)]
    
    nrows = len(state_chain_by_subject)
    fig, axes = plt.subplots(ncols=1, nrows=nrows)

    # legends
    state_patches = []
    for state_idx in range(n_states):
        patch = mpatches.Patch(
            color=state_colors[state_idx],
            label=('State ' + str(state_idx+1)))
        state_patches.append(patch)

    task_patches = []
    for task_idx, (key, color, ivals) in enumerate(task_annotations[0]):
        patch = mpatches.Patch(
            color=color,
            label=(str(key)))
        task_patches.append(patch)

    if nrows > 1:
        axes[0].add_artist(axes[0].legend(handles=state_patches, loc='upper left'))
        axes[0].legend(handles=task_patches, loc='upper right')
    else:
        axes.add_artist(axes.legend(handles=state_patches, loc='upper left'))
        axes.legend(handles=task_patches, loc='upper right')

    for row_idx in range(nrows):
        print("Drawing row " + str(row_idx+1))

        if nrows > 1:
            ax = axes[row_idx]
        else:
            ax = axes

        # set tick locations
        if tmax - tmin > 200:
            tick_ival = 100
        else:
            tick_ival = 20

        ticks = np.arange(tmin * sfreq, 
                          tmax * sfreq, 
                          tick_ival * sfreq)
        ax.set_xlim(tmin * sfreq,
                    tmax * sfreq)
        ax.set_xticks(ticks)
        ax.set_xticklabels(["%0.1f" % (tick / sfreq) 
                            for tick in ticks])

        ax.set_title(names[row_idx])

        state_chain = state_chain_by_subject[row_idx]

        # don't draw at all if tmin is larger than data from subject
        if tmin >= state_chain.shape[0] / sfreq:
            continue


        # find out first and last sample of the current subject
        start = int(tmin * sfreq)
        if tmax >= state_chain.shape[0] / sfreq:
            end = state_chain.shape[0]
        else:
            end = int(tmax * sfreq)

        print("Drawing state spans")
        for idx in range(start, end):
            if probabilistic:
                acc = [0] + (np.cumsum(state_chain[idx]) / 
                             np.cumsum(state_chain[idx])[-1]).tolist()
                ivals = [(acc[ii], acc[ii+1]) 
                         for ii in range(len(acc)-1)]
                for cidx, ival in enumerate(ivals):
                    ax.axvspan(idx, idx+1, ival[0], ival[1], alpha=0.7, 
                               color=state_colors[cidx], lw=0)
            else:
                cidx = np.argmax(state_chain[idx])
                ax.axvspan(idx, idx+1, alpha=0.7, 
                           color=state_colors[cidx], lw=0)

        if task_annotations:
            print("Drawing task spans")
            for key, color, ivals in task_annotations[row_idx]:
                for ival in ivals:

                    # don't draw if completely out of timeblock
                    if ival[1] < tmin:
                        continue
                    if ival[0] > tmax:
                        continue

                    # if partially out of time window, crop to data
                    if ival[0] <= tmin:
                        span_start = tmin
                    else:
                        span_start = ival[0]
                    if ival[1] >= tmax:
                        span_end = tmax
                    else:
                        span_end = ival[1]
                    
                    # draw a box to indicate task
                    span_start = int(span_start * sfreq)
                    span_end = int(span_end * sfreq)
                    ax.axvspan(span_start, span_end, 0.0, 0.2, alpha=1.0,
                               color=color, lw=0)

    fig.tight_layout()
    if show:
        plt.show()
    return fig 


def plot_task_comparison(chain, sfreq, task_times, fun, show=False, ylabel='', title=''):
    """
    """
    n_states = chain.shape[1]

    task_names = []
    task_vals = []
    task_cis = []
    for key, ivals in task_times:
        task_names.append(key)

        ival_chain = []
        for ival in ivals:
            start_idx = int(ival[0] * sfreq)
            end_idx = int(ival[1] * sfreq)
            ival_chain.append(chain[start_idx:end_idx])
        ival_chain = np.concatenate(ival_chain, axis=0)

        task_vals.append(fun(ival_chain, sfreq))

    labels = ['S' + str(ii+1).zfill(2) for ii in range(n_states)]
    
    x = np.arange(n_states)
    width = 0.3
    
    fig, ax = plt.subplots()

    locs = np.linspace(-width*len(task_vals) / 4, width * len(task_vals) / 4, len(task_vals))

    for idx in range(len(task_vals)):
        task_val = task_vals[idx]
        task_name = task_names[idx]
        loc = x + locs[idx]
        ax.bar(loc, task_val, width, label=task_name)

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    if show:
        plt.show()
    return fig 


def dwell_time(chain, sfreq):
    dwells = []
    for state_idx in range(chain.shape[1]):
        # first create a mask consisting of 1s and 0s
        mask = []
        for tidx in range(chain.shape[0]):
            if chain[tidx, state_idx] >= np.max(chain[tidx, :]):
                mask.append(1)
            else:
                mask.append(0)

        # then find consecutive ones
        lengths = []
        groups = groupby(mask)
        for key, group in groups:
            if key == 1:
                lengths.append(len(list(group)))

        dwells.append(np.mean(lengths)*sfreq)

    return dwells


def fractional_occupancy(chain, sfreq):
    frac_occ = np.mean(chain, axis=0)
    return frac_occ


def interval_length(chain, sfreq):
    intervals = []
    for state_idx in range(chain.shape[1]):
        # first create a mask consisting of 1s and 0s
        mask = []
        for tidx in range(chain.shape[0]):
            if chain[tidx, state_idx] >= np.max(chain[tidx, :]):
                mask.append(1)
            else:
                mask.append(0)

        # then find consecutive zeros
        lengths = []
        groups = groupby(mask)
        for key, group in groups:
            if key == 0:
                lengths.append(len(list(group)))

        intervals.append(np.mean(lengths)*sfreq)

    return intervals

