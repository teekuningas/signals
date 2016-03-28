import mne
import matplotlib.pyplot as plt
import numpy as np

from fourier_ica import FourierICA

if __name__ == '__main__':
    raw = mne.io.Raw('/home/zairex/Code/cibr/data/gradudemo/KH007_EOEC-pre.fif',
                     preload=True)
    fica = FourierICA(n_components=5, max_pca_components=30)
    fica.fit(raw)
