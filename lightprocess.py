import matplotlib.pyplot as plt
import mne
import numpy as np
import sys
import os

raw = mne.io.Raw(sys.argv[1], preload=True)

h_freq = 100

path = '/projects/meditaatio/workspace/maxfiltered/'

raw.filter(l_freq=None, h_freq=h_freq)

raw.resample(250)

raw.save(path + os.path.basename(sys.argv[1]), overwrite=True)
