# Authors: Erkka Heinila <erkka.heinila@jyu.fi>
#
# License: BSD (3-clause)

import numpy as np
import mne

from mne.time_frequency.stft import stft


class FourierICA(object):

    """M/EEG signal decomposition using STFT and Independent Component 
    Analysis (ICA)

    This object can be used to explore and filter interesting components
    found by ica used to short-time fourier transformed time series data.

    Parameters
    ----------
    ...
    """

    def __init__(self, wsize, sfreq, tstep=None, n_ica_components=None, n_pca_components=None, 
                 conveps=1e-7, maxiter=None, zerotolerance=None):
        self.wsize = wsize
        self.tstep = tstep
        self.n_pca_components = n_pca_components
        self.n_ica_components = n_ica_components
        self.zerotolerance = zerotolerance
        self.maxiter = maxiter
        self.conveps = conveps
        if not zerotolerance:
            self.zerotolerance = conveps

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
        pass

        # concatenate ft's to have two-dimensional data for ica
        fts = [stft_[:, :, idx] for idx in range(stft_.shape[2])]
        data2d = np.concatenate(fts, axis=1)

        # whiten the data
        whitened = self._whiten(data2d)

        # do ica
        components = self._fastica(whitened)

    def apply(self):
        pass

    def _fastica(self, data):
        """ 
        deflationary complex ica depcited from
        (Bingham and Hyvarinen, 2000)
        """

        if self.maxiter:
            maxiter = self.maxiter
        else:
            maxiter = max(40*self.n_pca_components, 2000)

        mixing_ = np.zeros((self.n_ica_components, self.n_pca_components), 
                           dtype=data.dtype)
        x = data

        for i in range(self.n_ica_components):

            # initial point, make it imaginary and length one
            r_ = np.random.randn(self.n_pca_components)
            i_ = np.random.randn(self.n_pca_components)
            w_old = r_ + 1j * i_
            w_old = w_old / np.linalg.norm(w_old)

            for j in range(maxiter/self.n_ica_components):

                # compute things
                y_ = np.dot(np.conj(w_old.T), x)
                g_ = np.log(1 + np.abs(y_)**2)
                dg_ = 1.0 / (1 + np.abs(y_)**2)
                first = np.mean(x*np.conj(y_)*g_, axis=1)
                second = np.mean(g_ + (np.abs(y_)**2)*dg_)*w_old

                # fixed-point iteration
                w_ = first - second

                # decorrelate
                projections = np.zeros(w_.shape)
                for k in range(i):
                    projections += np.dot(np.dot(mixing_[k], 
                                                 np.conj(mixing_[k].T), w_))
                w_ -= projections

                # renormalize
                w_ = w_ / np.linalg.norm(w_)

                # check if converged
                print w_

                # store old value
                w_old = w_

            mixing_[i, :] = w_

        return mixing_

    def _whiten(self, data):
        """
        Whiten data with PCA

        """

        # substract mean value from channels
        mean_ = data.mean(axis=-1)
        data -= mean_[:, np.newaxis]
        self.mean_ = mean_

        # calculate eigenvectors and eigenvalues from covariance matrix
        covmat = np.cov(data)
        eigw, eigv = np.linalg.eigh(covmat)

        # filter out components that are too small (or even negative)
        valids = np.where(eigw/eigw[-1] > self.zerotolerance)[0]
        eigw = eigw[valids]
        eigv = eigv[:, valids]

        # adjust number of components
        n_pca_components = self.n_pca_components
        if not n_pca_components:
            n_pca_components = len(valids)
        elif n_pca_components > len(valids):
            n_pca_components = len(valids)
        self.n_pca_components = n_pca_components

        # sort in descending order and take only n_pca_components of components
        eigw = eigw[::-1][0:n_pca_components]
        eigv = eigv[:, ::-1][:, 0:n_pca_components]

        # construct whitening matrix
        dsqrt = np.sqrt(eigw)
        dsqrtinv = 1.0/dsqrt
        whitening = np.dot(np.diag(dsqrtinv), np.conj(eigv.T))

        # whiten the data
        whitened = np.dot(whitening, data)

        return whitened

