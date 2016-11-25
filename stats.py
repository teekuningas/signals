import os
import sys

import pandas as pd
import numpy as np
from scipy import stats

PATH = sys.argv[-1]

files = os.listdir(PATH)
files = filter(lambda name: name.endswith('.csv'), files)

frames = [pd.read_csv(PATH + fname) for fname in files]

data = pd.concat(frames)
data.index = range(len(data))

import pdb; pdb.set_trace()
print "kissa"
