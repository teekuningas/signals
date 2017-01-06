import os
import sys

import pandas as pd
import numpy as np
from scipy import stats

filenames = sys.argv[1:]

frames = [pd.read_csv(fname) for fname in filenames]

data = pd.concat(frames)
data.index = range(len(data))

import pdb; pdb.set_trace()
print "kissa"
