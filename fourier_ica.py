# Authors: Erkka Heinila <erkka.heinila@jyu.fi>
#
# License: BSD (3-clause)

import numpy as np
import mne


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

    def __init__(self, n_components=None, max_pca_components=None):
        pass

    def fit(self, raw):
        pass

    def apply(self):
        pass
