import pickle
import sys
import os

import mne
import matplotlib.pyplot as plt
import numpy as np

from lib.load import load_layout
from lib.abstract import plot_components

MEG = False


def main():

    mne.utils.set_log_level('ERROR')

    if MEG:
        layout = None
    else:
        layout = load_layout()

    filenames = sys.argv[1:]

    components = []
    for fname in filenames:
        print "Opening " + fname
        part = pickle.load(open(fname, "rb"))
        components.extend(part)

    plot_components(components, layout)

    import pdb; pdb.set_trace()
    print "kissa"


if __name__ == '__main__':
    main()
