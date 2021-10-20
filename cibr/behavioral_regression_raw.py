import os
import argparse

import numpy as np
import sklearn
import mne
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso



parser = argparse.ArgumentParser()
parser.add_argument('--raw', nargs='+')
parser.add_argument('--questionnaire')
parser.add_argument('--save_path')

cli_args = parser.parse_args()


def identifier_from_path(path):
    fname = os.path.basename(path)
    if fname.startswith('IC'):
        identifier = 'I' + fname.split('_')[2][1:]
        return identifier
    else:
        identifier = 'M' + fname.split('_')[1]
        return identifier


questionnaire = []
with open(cli_args.questionnaire, 'r') as f:
    lines = f.readlines()
    header = lines[0].strip().split(',')
    for line in lines[1:]:
        questionnaire.append(line.strip().split(','))

# questionnaire = pd.DataFrame(questionnaire, columns=header)

def extract_band_power(path, band=(7, 14)):
    raw = mne.io.Raw(path, preload=True)
    picks = mne.pick_types(raw.info, meg=True, eeg=False)
    psds, freqs = mne.time_frequency.psd_welch(
        raw, tmin=raw.times[0]+5, tmax=raw.times[-1]-5,
        n_fft=int(raw.info['sfreq']*2))

    power = np.mean(psds[:, (freqs > band[0]) & (freqs < band[1])], axis=1)
    return power

X = []
for path in cli_args.raw:
    identifier = identifier_from_path(path)
    try:
        subj_idx = [row[0] for row in questionnaire].index(identifier)
    except ValueError:
        print(path + " skipped.")
        continue
    data = []
    data.append(extract_band_power(path, band=(1,4)))
    data.append(extract_band_power(path, band=(4,8)))
    data.append(extract_band_power(path, band=(8,13)))
    data.append(extract_band_power(path, band=(13,20)))
    data.append(extract_band_power(path, band=(20,40)))
    data.append(extract_band_power(path, band=(40,80)))
    X.append(np.concatenate(data, axis=0))

X = np.array(X)

# pre zscore
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

from sklearn.decomposition import PCA
pca = PCA(n_components=10, whiten=True)

X = pca.fit_transform(X)

print("Explained variance: " + str(np.sum(pca.explained_variance_ratio_)))

alphas = [0.01, 0.1, 1, 10, 100]
for behav in ['BIS', 'BDI', 'BAI', 'BasTotal']:
    print("Analysing " + behav)
    behav_idx = header.index(behav)

    y = []
    for path in cli_args.raw:
        identifier = identifier_from_path(path)
        try:
            subj_idx = [row[0] for row in questionnaire].index(identifier)
        except ValueError:
            print(path + " skipped.")
            continue

        y.append(int(questionnaire[subj_idx][behav_idx]))
    y = np.array(y)

    # X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    #               [3, 5, 7, 6, 11, 13, 7, 17, 19, 18, 24, 25]]).T
    # y = np.array([20, 18, 15, 14, 20, 10, 6, 6, 3, 2, 2, 1])

    test_scores = []
    train_scores = []
    for alpha in alphas:
        test_scores_cv = []
        train_scores_cv = []
        for idx in range(100):
            # train_idx, test_idx = list(KFold(n_splits=2, shuffle=True).split(X.copy()))[0]
            train_idx, test_idx = train_test_split(range(X.shape[0]), test_size=0.3)
            regr = Ridge(
                fit_intercept=True, alpha=alpha)
            regr.fit(X[train_idx], y[train_idx])
            train_scores_cv.append(regr.score(X[train_idx], y[train_idx]))
            test_scores_cv.append(regr.score(X[test_idx], y[test_idx]))

        test_scores.append((alpha, np.mean(test_scores_cv)))
        train_scores.append((alpha, np.mean(train_scores_cv)))

    print("Train scores")
    sorted_scores = sorted(train_scores, key=lambda x: x[1], reverse=True)
    from pprint import pprint
    pprint(sorted_scores[:20])

    print("Test scores")
    sorted_scores = sorted(test_scores, key=lambda x: x[1], reverse=True)
    from pprint import pprint
    pprint(sorted_scores[:20])


