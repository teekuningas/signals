# Authors: Erkka Heinila <erkka.heinila@jyu.fi>
#
# License: BSD (3-clause)

import numpy as np
import mne

from mne.time_frequency.stft import stft

from sklearn.decomposition import PCA


class FourierICA(object):

    """M/EEG signal decomposition using STFT and Independent Component 
    Analysis (ICA)

    This object can be used to explore and filter interesting components
    found by ica used to short-time fourier transformed time series data.

    Parameters
    ----------
    n_components : int | float | None
        The number of components used for ICA decomposition. If int, it must be
        smaller then max_pca_components. If None, all PCA components will be
        used. If float between 0 and 1 components can will be selected by the
        cumulative percentage of explained variance.
    max_pca_components : int | None
        The number of components used for PCA decomposition. If None, no
        dimension reduction will be applied and max_pca_components will equal
        the number of channels supplied on decomposing data. Defaults to None.
    """

    def __init__(self, wsize, sfreq, tstep=None, n_ica_components=None, n_pca_components=None, 
                 zerotolerance=1e-14, lpass=1, hpass=80):
        self.wsize = wsize
        self.tstep = tstep
        self.n_pca_components = n_pca_components
        self.n_ica_components = n_ica_components
        self.lpass = lpass
        self.hpass = hpass
        self.zerotolerance = zerotolerance

    def fit(self, data):
        """ Fit data

        Parameters
        ---------
        data : ndarray, shape (channels, times)

        Returns
        ------

        """
        # first do stft
        stft_ = stft(data, self.wsize, self.tstep)
        # remove outliers

        # concatenate ft's to have two-dimensional data for ica
        fts = [stft_[:, :, idx] for idx in range(stft_.shape[2])]
        data2d = np.concatenate(fts, axis=1)

        whitened = self._whiten(data2d)

        import pdb; pdb.set_trace()

        # do ica

        # sort components

    def apply(self):
        pass

    def _whiten(self, data):
        """ Whiten data with PCA
        """

        # substract mean value from channels
        mean_ = data.mean(axis=-1)
        data -= mean_[:, np.newaxis]

        # calculate eigenvectors and eigenvalues from covariance matrix
        covmat = np.cov(data)
        covmat = np.conj(covmat)
        eigw, eigv = np.linalg.eigh(covmat)

        # filter out components that are too small (or even negative)
        valids = np.where(eigw > self.zerotolerance)[0]
        eigw = eigw[valids]
        eigv = eigv[:, valids]

        # adjust number of copmonents
        n_pca_components = self.n_pca_components
        if not n_pca_components:
            n_pca_components = len(valids)
        elif n_pca_components > len(valids):
            n_pca_components = len(valids)

        # sort in descending order and take only n_pca_components of components
        eigw = eigw[::-1][0:n_pca_components]
        eigv = eigv[:, ::-1][:, 0:n_pca_components]

        # construct whitening and dewhitening matrices
        dsqrt = np.sqrt(eigw)
        dsqrtinv = 1.0/dsqrt
        self.whitening = np.dot(np.diag(dsqrtinv), eigv.T)
        self.dewhitening = np.dot(eigv, np.diag(dsqrt))

        # whiten the data
        whitened = np.dot(self.whitening, data)
        return whitened
