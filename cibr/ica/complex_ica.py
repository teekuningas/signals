# Authors: Erkka Heinila <erkka.heinila@jyu.fi>
#
# License: BSD (3-clause)
import sys

import numpy as np
import mne

from scipy.linalg import sqrtm
from numpy.linalg import inv


def _fastica(data, n_components, maxiter, conveps, random_state):
    """ 
    Complex fastica depicted from
    (Bingham and Hyvarinen, 2000)
    """

    z = data

    def sym_decorrelation(w_):
        return np.dot(w_, sqrtm(inv(np.dot(np.conj(w_.T), w_))))

    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    # get decorrelated initial mixing matrix
    if random_state:
        r_ = random_state.randn(n_components, n_components)
        i_ = random_state.randn(n_components, n_components)
    else:
        r_ = np.random.randn(n_components, n_components)
        i_ = np.random.randn(n_components, n_components)

    w_old = r_ + 1j * i_
    w_old = sym_decorrelation(w_old)

    for j in range(maxiter):

        # get new mixing matrix by updating columns one by one
        w_new = np.zeros((n_components, n_components), 
                         dtype=np.complex128)
        for i in range(w_old.shape[1]):
            y_ = np.dot(np.conj(w_old[:, i]).T, z)
        
            g_ = 1.0/(0.1 + np.abs(y_)**2)
            dg_ = -1.0 / (0.1 + np.abs(y_)**2)**2

            first = np.mean(z*np.conj(y_)*g_, axis=-1)
            second = np.mean(g_ + (np.abs(y_)**2)*dg_, axis=-1)*w_old[:, i]
            w_new[:, i] = first - second

        # symmetrically decorrelate
        w_new = sym_decorrelation(w_new)

        # calculate convergence criterion
        criterion = (np.sum(np.abs(np.sum(w_new*np.conj(w_old), axis=1))) /
                     n_components)

        # show something
        if j%30 == 0:
            print "Criterion:", str(1 - criterion), 
            print "- Conveps:", str(conveps),
            print "- i:", str(j), "- maxiter:", str(maxiter) 
            sys.stdout.flush()

        # check if converged
        if 1 - criterion < conveps:
            y_ = np.dot(np.conj(w_new.T), z)
            g_ = np.log(1 + np.abs(y_)**2)
            print '\nObjective values are: ' + str(np.mean(g_, axis=-1))
            print 'Convergence value: ' + str(1 - criterion)
            break

        # store old value
        w_old = w_new

    if j+1 == maxiter:
        raise Exception('ICA did not converge.')

    print 'ICA finished with ' + str(j+1) + ' iterations'

    return w_new, np.dot(np.conj(w_new).T, z)

def _whiten(data, zerotolerance, n_components):
    """
    Whiten data with PCA

    """

    # calculate covariance matrix
    covmat = np.cov(data)

    # calculate eigenvectors and eigenvalues from covariance matrix
    eigw, eigv = np.linalg.eigh(covmat)

    # filter out components that are too small (or even negative)
    valids = np.where(eigw/eigw[-1] > zerotolerance)[0]
    eigw = eigw[valids]
    eigv = eigv[:, valids]

    # adjust number of pca components
    if n_components > len(valids):
        n_components = len(valids)

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

    return n_components, whitening, dewhitening, whitened


def complex_ica(data, n_components, conveps=1e-7, maxiter=2000, zerotolerance=1e-7,
                random_state=None):
    """
    solves y = (W^H)Vx, where 
      x is mixed multidimensional time series,
      V is whitening matrix,
      W is mixing matrix, and
      y contains the unmixed sources

    Parameters:
    n_components: number of estimated components,
    conveps: when to stop iterating,
    maxiter: maximum number of iterations,
    zerotolerance: how close to zero can the smallest eigenvalues be when whitening. If normalized eigenvalues go under zerotolerance, number of components is decreased so that all eigenvalues are over the zerotolerance.

    Returns:
    sorted_ic: independent components sorted by the nongaussianity score
    sorted_mixing: mixing matrix where columns are sorted by the nongaussianity score
    dewhitening: dewhitening matrix
    unmixing: unmixing matrix
    whitening: whitening matrix
    mean: mean before whitening
    """

    # substract mean value from channels
    data = data.copy()
    mean = data.mean(axis=-1)
    data -= mean[:, np.newaxis]

    print "Whiten data"
    n_components, whitening, dewhitening, whitened = _whiten(data, zerotolerance, n_components)
    
    print "Do ICA"
    mixing_, ic_ = _fastica(whitened, n_components, maxiter, conveps, random_state)

    # sort according to objective value
    objectives = []
    for i in range(ic_.shape[0]):
        component = ic_[i, :]
        g_ = np.log(1 + np.abs(component)**2)
        objectives.append(np.mean(g_))
    indices = np.argsort(objectives)[::-1]

    sorted_ic = ic_[np.argsort(objectives), :]
    sorted_mixing = mixing_[:, np.argsort(objectives)]

    return sorted_ic, sorted_mixing, dewhitening, np.conj(sorted_mixing).T, whitening, mean


class ComplexICA(object):
    def __init__(self, n_components=20, conveps=1e-6, maxiter=2000, 
                 zerotolerance=1e-7, random_state=None):
        """
        """
        self.n_components = n_components
        self.conveps = conveps
        self.maxiter = maxiter
        self.zerotolerance = zerotolerance
        self.random_state = random_state

    def fit(self, data):
        sources, mixing, dewhitening, unmixing, whitening, mean = (
            complex_ica(data,
                        self.n_components,
                        self.conveps,
                        self.maxiter,
                        self.zerotolerance,
                        self.random_state))
        self.unmixing = unmixing
        self.mixing = mixing
        self.whitening = whitening
        self.dewhitening = dewhitening
        self.sources = sources
