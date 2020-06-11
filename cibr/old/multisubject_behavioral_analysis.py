PLOT_TO_PICS=True

if PLOT_TO_PICS:
    import matplotlib
    matplotlib.rc('figure', max_open_warning=0)
    matplotlib.use('Agg')

import pyface.qt

import sys
import argparse
import os

import mne
import scipy
import scipy.sparse
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import matplotlib as mpl
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

from signals.cibr.lib.stc import plot_vol_stc_brainmap
from signals.cibr.lib.sensor import plot_sensor_topomap
from signals.cibr.lib.utils import MidpointNormalize

from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path')
    parser.add_argument('--questionnaire')
    parser.add_argument('--behav_measure')
    parser.add_argument('--identifier')
    parser.add_argument('--drop')
    parser.add_argument('--example_raw')
    parser.add_argument('--coefficients_1', nargs='+')
    parser.add_argument('--coefficients_2', nargs='+')

    cli_args = parser.parse_args()

    subjects_dir = os.environ['SUBJECTS_DIR']

    save_path = cli_args.save_path if PLOT_TO_PICS else None

    data_1 = []
    data_2 = []
    names_1 = []
    names_2 = []
    vertex_list = []

    # read contrast data from all participants
    drop_keys = cli_args.drop.split(' ') if cli_args.drop else []

    for fname in sorted(cli_args.coefficients_1):
        if any([key in fname for key in drop_keys]):
            continue

        with open(fname, 'r') as f:
            lines = f.readlines()
            vertex_list.append([val for val in lines[0].strip().split(', ')])
            data_1.append([float(val) for val in lines[1].split(', ')])
            names_1.append(fname.split('/')[-1].split('_')[1])
    data_1 = np.array(data_1)

    for fname in sorted(cli_args.coefficients_2):
        if any([key in fname for key in drop_keys]):
            continue

        with open(fname, 'r') as f:
            lines = f.readlines()
            data_2.append([float(val) for val in lines[1].split(', ')])
            names_2.append(fname.split('/')[-1].split('_')[1])
    data_2 = np.array(data_2)

    if names_1 != names_2:
        raise Exception('Names do not match')
    names = names_1

    contrast_data = np.mean([data_1, data_2], axis=0)
    norm_data = np.abs(contrast_data)

    # for plotting
    raw = mne.io.Raw(cli_args.example_raw, preload=True)
    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names'])
                       if idx not in mne.pick_types(raw.info, meg=True)])

    # read behavioral questionnaire data
    questionnaire = []
    with open(cli_args.questionnaire, 'r') as f:
        lines = f.readlines()
        header = (['id'] + [elem.strip('\n') for elem in 
                           lines[0][1:].split(',')])
        for line in lines[1:]:
            elems = [elem.strip('\n') for elem in line.split(',')]
            questionnaire.append([elems[0].zfill(3)] + elems[1:])

    questionnaire = pd.DataFrame(questionnaire, columns=header)

    behav_measure = cli_args.behav_measure

    behavs = []
    for name in names:
        behavs.append(
            float(questionnaire[questionnaire['id'] == name][behav_measure].values[0]))

    ## VERTEXWISE CLUSTER ANAlySIS

    if set(behavs) == set([0, 1]):
        stat_corr = False
    else:
        stat_corr = True

    corr_fun = lambda a, b: scipy.stats.spearmanr(a, b)[0]
    # corr_fun = lambda a, b: scipy.stats.pearsonr(a, b)[0]

    # corr_weights = np.ones(norm_data.shape[1])
    # corr_weights = (np.std(norm_data, axis=0) / np.max(np.std(norm_data, axis=0)))**(0.2)
    corr_weights = (np.std(norm_data, axis=0) / np.max(np.std(norm_data, axis=0)))**(0.2)

    if stat_corr:
        statmap = []
        for vert_idx in range(contrast_data.shape[1]):
            Y = contrast_data[:, vert_idx]
            X = behavs
            corrcoef = corr_fun(X, Y)
            statmap.append(corrcoef)
        # statmap = np.array(statmap)
        statmap = np.array(statmap * corr_weights)

        # crit_t_val = scipy.stats.t.ppf(0.975, df=contrast_data.shape[0]-2)
        # crit_t_val = scipy.stats.t.ppf(0.95, df=contrast_data.shape[0]-2)
        crit_t_val = scipy.stats.t.ppf(0.9, df=contrast_data.shape[0]-2)
        crit_val = np.sqrt(crit_t_val**2 / (crit_t_val**2 + contrast_data.shape[0] - 2))

        def stat_fun(x):
            corrcoefs = np.array([corr_fun(x[:, idx], behavs) * corr_weights[idx] 
                                  for idx in range(x.shape[1])])
            return corrcoefs
    else:
        statmap = []
        for vert_idx in range(contrast_data.shape[1]):
            subdata = contrast_data[:, vert_idx]
            group = np.array(behavs)
            statmap.append(scipy.stats.ttest_ind(subdata[group==1], subdata[group==0])[0])
        statmap = np.array(statmap)

        crit_val = scipy.stats.t.ppf(0.975, df=contrast_data.shape[0]-2)

        def stat_fun(x):
            group = np.array(behavs)
            statmap = []
            for vert_idx in range(x.shape[1]):
                subdata = x[:, vert_idx]
                statmap.append(scipy.stats.ttest_ind(subdata[group==1], subdata[group==0])[0])
            return np.array(statmap)

    print("Critical value: " + str(crit_val))

    connectivity = scipy.sparse.coo_matrix(np.corrcoef(norm_data.T))
    results = mne.stats.cluster_level.spatio_temporal_cluster_1samp_test(
        contrast_data[:, np.newaxis, :],
        connectivity=connectivity,
        n_permutations=1024,
        out_type='indices',
        threshold=crit_val,
        stat_fun=stat_fun)

    clusters = []
    for cluster_idx in range(len(results[2])):
        # add jitter to avoid overlay bug
        cluster_map = np.array([np.abs(np.random.normal(scale=0.001)) 
                                for _ in range(contrast_data.shape[1])])
        for vert_idx in results[1][cluster_idx][1]:
            cluster_map[vert_idx] = 1 + np.random.normal(scale=0.001)
        
        cluster_verts = results[1][cluster_idx][1]
        pvalue = results[2][cluster_idx]

        if pvalue > 0.15:
            continue

        clusters.append((pvalue, cluster_verts, cluster_map))

    n_clusters = len(clusters)

    fig = plt.figure()
    ax_statmap = plt.subplot2grid((2+5*n_clusters, 1), (0, 0), rowspan=2)
    ax_clusters = []
    for cluster_idx in range(n_clusters):
        ax_empty = plt.subplot2grid((2+5*n_clusters, 1), (2+5*cluster_idx + 0, 0), rowspan=1)
        ax_map = plt.subplot2grid((2+5*n_clusters, 1), (2+5*cluster_idx + 1, 0), rowspan=2)
        ax_detail = plt.subplot2grid((2+5*n_clusters, 1), (2+5*cluster_idx + 3, 0), rowspan=2)
        ax_clusters.append((ax_empty, ax_map, ax_detail))

    if len(statmap) > 500:
        vertices = np.array([int(vx) for vx in vertex_list[0]])
        plot_vol_stc_brainmap(statmap, vertices, '10', subjects_dir,
                              ax_statmap)
        for cluster_idx in range(n_clusters):
            ax_empty = ax_clusters[cluster_idx][0]
            ax_map = ax_clusters[cluster_idx][1]
            ax_detail = ax_clusters[cluster_idx][2]
            
            ax_empty.axis('off')

            plot_vol_stc_brainmap(clusters[cluster_idx][2], vertices,
                                  '10', subjects_dir, ax_map,
                                  cmap='PiYG')

            cluster_vert_idxs = clusters[cluster_idx][1]
            vert_idx = np.argmax(np.abs(statmap)[cluster_vert_idxs])
            if stat_corr:
                X = contrast_data[:, cluster_vert_idxs][:, vert_idx]
                X = X / np.std(X)
                Y = behavs
                frame = pd.DataFrame(np.transpose([X, Y]), 
                                     columns=['Brain component', 'Behavioral component'])
                sns.regplot(x='Brain component', y='Behavioral component',
                            data=frame, ax=ax_detail, scatter_kws={'s':5}, line_kws={'lw': 1})
                for idx in range(X.shape[0]):
                    ax_detail.annotate(names[idx], (X[idx], Y[idx]), fontsize=6)

                ax_detail.set_title('regplot for cluster ' + str(cluster_idx+1))
            else:
                subdata = contrast_data[:, cluster_vert_idxs][:, vert_idx]
                group = np.array(behavs)
                data_1 = subdata[group==0]
                data_2 = subdata[group==1]

                frame = pd.DataFrame(
                    np.transpose([np.concatenate([data_1, data_2]), 
                                  np.concatenate([[0]*len(data_1), [1]*len(data_2)])]), 
                    columns=['Power', 'Group'])

                sns.boxplot(x="Group", y="Power", data=frame, ax=ax_detail)

                ax_detail.set_title('boxplots for cluster ' + str(cluster_idx+1))

            ax_map.set_title(
                '\n\n' + 'cluster ' + str(cluster_idx+1) + ' (pvalue ' + 
                             str(clusters[cluster_idx][0]) + ')')

    else:
        plot_sensor_topomap(statmap, raw.info, ax_statmap)
        for cluster_idx in range(n_clusters):
            ax_empty = ax_clusters[cluster_idx][0]
            ax_map = ax_clusters[cluster_idx][1]
            ax_detail = ax_clusters[cluster_idx][2]

            ax_empty.axis('off')

            plot_sensor_topomap(clusters[cluster_idx][2], raw.info,
                                ax_map, cmap='PiYG')

            cluster_vert_idxs = clusters[cluster_idx][1]
            vert_idx = np.argmax(np.abs(statmap)[cluster_vert_idxs])
            if stat_corr:
                X = contrast_data[:, cluster_vert_idxs][:, vert_idx]
                X = X / np.std(X)
                Y = behavs
                frame = pd.DataFrame(np.transpose([X, Y]), 
                                     columns=['Brain component', 'Behavioral component'])
                sns.regplot(x='Brain component', y='Behavioral component',
                            data=frame, ax=ax_detail, scatter_kws={'s':5}, line_kws={'lw': 1})
                for idx in range(X.shape[0]):
                    ax_detail.annotate(names[idx], (X[idx], Y[idx]), fontsize=6)

                ax_detail.set_title('regplot for cluster ' + str(cluster_idx+1))
            else:
                subdata = contrast_data[:, cluster_vert_idxs][:, vert_idx]
                group = np.array(behavs)
                data_1 = subdata[group==0]
                data_2 = subdata[group==1]

                frame = pd.DataFrame(
                    np.transpose([np.concatenate([data_1, data_2]), 
                                  np.concatenate([[0]*len(data_1), [1]*len(data_2)])]), 
                    columns=['Power', 'Group'])

                sns.boxplot(x="Group", y="Power", data=frame, ax=ax_detail)

                ax_detail.set_title('boxplots for cluster ' + str(cluster_idx+1))

            ax_map.set_title('cluster ' + str(cluster_idx+1) + ' (pvalue ' + 
                             str(clusters[cluster_idx][0]) + ')')

    cmap = mpl.cm.RdBu_r

    divider = make_axes_locatable(ax_statmap)
    ax_cbar = divider.append_axes("right", size="2%", pad=0.0)
    norm = MidpointNormalize(np.min(statmap), np.max(statmap))
    cb = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap,
        norm=norm,
        orientation='vertical')

    if stat_corr:
        ax_statmap.set_title('Correlation (' + str(behav_measure) + ')')
    else:
        ax_statmap.set_title('Independent t tests (' + str(behav_measure) + ')')

    if save_path:
        brain_path = os.path.join(save_path, 'brainmaps')
        if not os.path.exists(brain_path):
            os.makedirs(brain_path)
        fname = ('vertexwise_' + str(cli_args.identifier) + '_' + 
                 str(behav_measure).lower() + '.png')
        fig.set_size_inches(6, 10)
        fig.savefig(os.path.join(brain_path, fname), dpi=310)

    raise Exception('Kissa')

    print("Computing normal PCA ..")

    n_brain_components = 5

    brain_total_variance = np.sum(np.diag(np.cov(data.T)))

    brain_pca = PCA(n_components=n_brain_components, whiten=False)
    brain_pca_comps = brain_pca.fit_transform(data)
    brain_pca_explained = np.diag(np.cov(brain_pca_comps.T)) / brain_total_variance
    print("PCA explained variance: " + str(brain_pca_explained))
    print("Sum: " + str(np.sum(brain_pca_explained)))

    # do validation by comparing pca's without one participant to the 
    # main one with all participants
    for idx in range(data.shape[0]):
        test_data = np.concatenate([data[:idx], data[idx+1:]])
        test_pca = PCA(n_components=n_brain_components, whiten=False)
        test_comps = test_pca.fit_transform(test_data)

        # correlation with main first pca component
        main_mix_first = np.linalg.pinv(brain_pca.components_)[:, 0]
        test_mix = np.linalg.pinv(test_pca.components_)
        coeffs = []
        for ii in range(n_brain_components):
            coeffs.append(np.abs(np.corrcoef(main_mix_first, test_mix[:, ii])[0, 1]))
        coeffs = np.array(coeffs)

        test_total_variance = np.sum(np.diag(np.cov(test_data.T)))
        test_pca_explained = np.diag(np.cov(test_comps.T)) / test_total_variance

        print("Results without participant " + str(names[idx]) + ": ")
        print("Explained variance " + str(names[idx]) + ": " + str(test_pca_explained))
        print("Sorted correlations: " + str(-np.sort(-coeffs)))
        print("Correlation order: " + str(np.argsort(-coeffs)))

    # scree plot
    fig, ax = plt.subplots()
    ax.plot(range(1, len(brain_pca_explained) + 1, 1), brain_pca_explained)
    ax.set_xlabel('Component')
    ax.set_ylabel('Explained variance')
    if save_path:
        stats_path = os.path.join(save_path, 'stats')
        if not os.path.exists(stats_path):
            os.makedirs(stats_path)
        fig.savefig(os.path.join(stats_path, 'scree_plot.png'))

    # continue analysis, start by ensuring that component variances are 1
    factors = np.std(brain_pca_comps, axis=0)
    brain_pca_comps = np.divide(brain_pca_comps, factors)
    brain_pca_mixing = np.multiply(np.linalg.pinv(brain_pca.components_), factors)
    brain_pca_unmixing = np.divide(brain_pca.components_.T, factors).T

    # convert from mainly blue to mainly red always
    for idx in range(n_brain_components):
        if np.mean(brain_pca_mixing[:, idx]) < 0:
            brain_pca_mixing[:, idx] = -brain_pca_mixing[:, idx]
            brain_pca_unmixing[idx, :] = -brain_pca_unmixing[idx, :]
            brain_pca_comps[:, idx] = -brain_pca_comps[:, idx]

    # plot mixing matrices overlayed to brain
    for idx in range(brain_pca_mixing.shape[1]):
        fig, ax_brain = plt.subplots()

        brain_map = brain_pca_mixing[:, idx]

        if len(brain_map) > 500:
            vertices = np.array([int(vx) for vx in vertex_list[0]])
            plot_vol_stc_brainmap(brain_map, vertices, '10', subjects_dir,
                                  ax_brain)
        else:
            plot_sensor_topomap(brain_map, raw.info, ax_brain)

        cmap = mpl.cm.RdBu_r

        divider = make_axes_locatable(ax_brain)
        ax_cbar = divider.append_axes("right", size="2%", pad=0.0)
        norm = MidpointNormalize(np.min(brain_map), np.max(brain_map))
        cb = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap,
            norm=norm,
            orientation='vertical')

        ax_brain.set_title('PCA component ' + str(idx+1))

        if save_path:
            brain_path = os.path.join(save_path, 'brainmaps')
            if not os.path.exists(brain_path):
                os.makedirs(brain_path)
            fname = ('pca_' + str(cli_args.identifier) + '_' + 
                     str(idx+1).zfill(2) + '.png')
            fig.savefig(os.path.join(brain_path, fname), dpi=310)

    # now the regression for the first component
    Y = np.array(behavs)
    X = np.array(brain_pca_comps[:, 0])

    model = sm.OLS(Y, sm.add_constant(X))
    results = model.fit()

    print("Results for brain PCA: ")
    print(results.summary())

    ## here need scatter
    sns.set(style='ticks', font_scale=1.2, font='Arial')
    fig, ax = plt.subplots()
    frame = pd.DataFrame(np.transpose([X, Y]), 
                         columns=['Brain component', 'Behavioral component'])
    sns.regplot(x='Brain component', y='Behavioral component',
                data=frame, ax=ax)

    if save_path:
        scatter_path = os.path.join(save_path, 'scatters')
        if not os.path.exists(scatter_path):
            os.makedirs(scatter_path)
        fig.savefig(os.path.join(scatter_path, 'scatter.png'))

    raise Exception('KISSA')

    ## ICA

    print("Computing ICA..")

    n_brain_components = 5

    brain_ica = FastICA(n_components=n_brain_components)
    brain_ica_comps = brain_ica.fit_transform(data)
    brain_ica_mixing = np.linalg.pinv(brain_ica.components_)

    # plot mixing matrices overlayed to brain
    for idx in range(brain_ica_mixing.shape[1]):
        fig, ax = plt.subplots()
        plot_vol_stc_brainmap(None, '',
                              brain_ica_mixing[:, idx], vertices, '10', subjects_dir,
                              axes=ax)
        fig.suptitle('ICA component ' + str(idx+1))
        if save_path:
            brain_path = os.path.join(save_path, 'brainmaps')
            if not os.path.exists(brain_path):
                os.makedirs(brain_path)
            path = os.path.join(brain_path, 'brain_ica_' + str(idx+1).zfill(2) + '.png')
            fig.savefig(path, dpi=310)

    for idx in range(brain_ica_comps.shape[1]):
        print("Regression for ICA component " + str(idx+1) + ": ")
        Y = behavs
        X = np.array(brain_ica_comps[:, idx])
        model = sm.OLS(Y, sm.add_constant(X))
        results = model.fit()
        print(results.summary())

    ## SPATIAL ICA
    
    print("Computing Spatial ICA..")

    n_brain_components = 5

    brain_ica = FastICA(n_components=n_brain_components)
    brain_ica_comps = brain_ica.fit_transform(data.T)
    brain_ica_mixing = np.linalg.pinv(brain_ica.components_)

    # plot mixing matrices overlayed to brain
    for idx in range(brain_ica_comps.shape[1]):
        fig, ax = plt.subplots()
        plot_vol_stc_brainmap(None, '',
                              brain_ica_comps[:, idx], vertices, '10', subjects_dir,
                              axes=ax)
        fig.suptitle('ICA component ' + str(idx+1))
        if save_path:
            brain_path = os.path.join(save_path, 'brainmaps')
            if not os.path.exists(brain_path):
                os.makedirs(brain_path)
            path = os.path.join(brain_path, 'brain_spatialica_' + str(idx+1).zfill(2) + '.png')
            fig.savefig(path, dpi=310)

    for idx in range(brain_ica_mixing.shape[1]):
        print("Regression for Spatial ICA component " + str(idx+1) + ": ")
        Y = behavs
        X = np.array(brain_ica_mixing[:, idx])
        model = sm.OLS(Y, sm.add_constant(X))
        results = model.fit()
        print(results.summary())

    raise Exception('MIAU')

