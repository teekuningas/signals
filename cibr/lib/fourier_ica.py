# Authors: Erkka Heinila <erkka.heinila@jyu.fi>
#
# License: BSD (3-clause)
import sys

import numpy as np

from mne.time_frequency.stft import stft
from mne.time_frequency.stft import stftfreq

from scipy.linalg import sqrtm
from numpy.linalg import inv


class FourierICA(object):

    """M/EEG signal decomposition using STFT and Independent Component 
    Analysis (ICA)

    This object can be used to explore and filter interesting components
    found by ica used to short-time fourier transformed time series data.

    Parameters
    ----------
    ...
    """

    def __init__(self, stft, freqs, n_components, conveps=None, maxiter=None, 
                 zerotolerance=None):
        self.stft = stft
        self._freqs = freqs
        self.n_components = n_components
        self.conveps = conveps
        self.maxiter = maxiter
        self.zerotolerance = zerotolerance

    def fit(self):
        """ Fit data

        Parameters
        ---------
        data : ndarray, shape (channels, freqs, times)

        """

        stft_ = self.stft

        # store shape to retrieve it later
        self._stft_shape = stft_.shape

        # concatenate data 
        data2d = self._concat(stft_)

        print "Whiten data"
        dewhitening, whitened = self._whiten(data2d)
        
        print "Do ICA"
        mixing_, ic_ = self._fastica(whitened)

        # sort according to objective value
        objectives = []
        for i in range(ic_.shape[0]):
            component = ic_[i, :]
            g_ = np.log(1 + np.abs(component)**2)
            objectives.append(np.mean(g_))
        indices = np.argsort(objectives)[::-1]

        sorted_ic = ic_[np.argsort(objectives), :]
        sorted_mixing = mixing_[:, np.argsort(objectives)]

        # store for retrieving
        self._mixing = sorted_mixing
        self._dewhitening = dewhitening
        self._source_stft = self._split(sorted_ic)

    @property
    def source_stft(self):
        return self._source_stft

    @property
    def freqs(self):
        return self._freqs

    def component_in_sensor_space(self, idx):
        """
        """
        # get concatenated source stft
        data = self._concat(self._source_stft)

        # zero out other components 
        data[:idx, :] = 0
        data[idx+1:, :] = 0

        # use mixing matrix to get to whitened sensor space
        data = np.dot(self._mixing, data)

        # dewhiten
        data = np.dot(self._dewhitening, data)

        # add the mean
        data += self._mean[:, np.newaxis]

        # split again and return
        return self._split(data)

    def _fastica(self, data):
        """ 
        Complex fastica depicted from
        (Bingham and Hyvarinen, 2000)
        """

        if self.maxiter:
            maxiter = self.maxiter
        else:
            maxiter = max(200 * self.n_components, 2000)

        if self.conveps:
            conveps = self.conveps
        else:
            conveps = 1e-13

        x = data

        def sym_decorrelation(w_):
            return np.dot(w_, sqrtm(inv(np.dot(np.conj(w_.T), w_))))

        # get decorrelated initial mixing matrix
        r_ = np.random.randn(self.n_components, self.n_components)
        i_ = np.random.randn(self.n_components, self.n_components)
        w_old = r_ + 1j * i_
        w_old = sym_decorrelation(w_old)

        for j in range(maxiter):

            # get new mixing matrix by updating columns one by one
            w_new = np.zeros((self.n_components, self.n_components), 
                             dtype=np.complex128)
            for i in range(w_old.shape[1]):
                y_ = np.dot(np.conj(w_old[:, i]).T, x)
            
                g_ = 1.0/(0.1 + np.abs(y_)**2)
                dg_ = -1.0 / (0.1 + np.abs(y_)**2)**2

                first = np.mean(x*np.conj(y_)*g_, axis=-1)
                second = np.mean(g_ + (np.abs(y_)**2)*dg_, axis=-1)*w_old[:, i]  # noqa
                w_new[:, i] = first - second

            # symmetrically decorrelate
            w_new = sym_decorrelation(w_new)

            # calculate convergence criterion
            criterion = (np.sum(np.abs(np.sum(w_new*np.conj(w_old), axis=1))) /
                         self.n_components)

            # show something
            if j%30 == 0:
                print "Criterion:", str(1 - criterion), 
                print "- Conveps:", str(conveps),
                print "- i:", str(j), "- maxiter:", str(maxiter) 
                sys.stdout.flush()

            # check if converged
            if 1 - criterion < conveps:
                y_ = np.dot(np.conj(w_new.T), x)
                g_ = np.log(1 + np.abs(y_)**2)
                print '\nObjective values are: ' + str(np.mean(g_, axis=-1))
                print 'Convergence value: ' + str(1 - criterion)
                break

            # store old value
            w_old = w_new

        if j+1 == maxiter:
            raise Exception('ICA did not converge.')

        print 'ICA finished with ' + str(j+1) + ' iterations'

        return w_new, np.dot(np.conj(w_new).T, x)

    def _whiten(self, data):
        """
        Whiten data with PCA

        """
        # substract mean value from channels
        mean_ = data.mean(axis=-1)
        data -= mean_[:, np.newaxis]
        self._mean = mean_

        # calculate covariance matrix
        covmat = np.cov(data)

        # calculate eigenvectors and eigenvalues from covariance matrix
        eigw, eigv = np.linalg.eigh(covmat)

        # filter out components that are too small (or even negative)
        if not self.zerotolerance:
            self.zerotolerance = 1e-7

        valids = np.where(eigw/eigw[-1] > self.zerotolerance)[0]
        eigw = eigw[valids]
        eigv = eigv[:, valids]

        # adjust number of pca components
        n_components = self.n_components
        if not n_components:
            n_components = len(valids)
        elif n_components > len(valids):
            n_components = len(valids)
        self.n_components = n_components

        # sort in descending order and take only n_components of components
        eigw = eigw[::-1][0:n_components]
        eigv = eigv[:, ::-1][:, 0:n_components]

        # construct whitening matrix
        dsqrt = np.sqrt(eigw)
        dsqrtinv = 1.0/dsqrt
        whitening = np.dot(np.diag(dsqrtinv), np.conj(eigv.T))

        # whiten the data, note no transpose
        whitened = np.dot(whitening, data)

        # dewhitening matrix
        dewhitening = np.dot(eigv, np.diag(dsqrt))

        return dewhitening, whitened

    def _concat(self, data):
        """
        """
        fts = [data[:, :, idx] for idx in range(data.shape[2])]
        data2d = np.concatenate(fts, axis=1)
        return data2d

    def _split(self, data):
        """
        """
        parts = np.split(data, self._stft_shape[2], axis=1)

        xw = data.shape[0]
        yw = data.shape[1]/self._stft_shape[2]
        zw = self._stft_shape[2]

        splitted = np.empty((xw, yw, zw), dtype=data.dtype)
        for idx, part in enumerate(parts):
            splitted[:, :, idx] = part

        return splitted


def fourier_ica_from_raw(raw, wsize, n_components, tstep=None, 
                 conveps=None, maxiter=None, zerotolerance=None, 
                 lpass=None, hpass=None, sfreq=None):

    print "First do stft"
    stft_ = stft(raw._data, wsize, tstep)

    # bandpass filter
    if sfreq:
        freqs = stftfreq(wsize, sfreq)

        hpass_, lpass_ = 0, len(freqs)
        if hpass:
            hpass_ = min(np.where(freqs >= hpass)[0])
        if lpass:
            lpass_ = max(np.where(freqs <= lpass)[0])

        self._freqs = freqs[hpass_:lpass_]

        stft_ = stft_[:, hpass_:lpass_, :]

    fica = FourierICA(raw._data, freqs, n_components, conveps=None, maxiter=None, 
                      zerotolerance=None, lpass=None, hpass=None, sfreq=None)

    return fica


if __name__ == '__main__':
    print "This script is not runnable, used only as a library"
