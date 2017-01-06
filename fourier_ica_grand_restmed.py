import sys
import pickle

import mne
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from lib.fourier_ica import FourierICA
from lib.component import ComponentPlot
from lib.load import load_layout


STEP_SIZE = 1024.0
SAVE_PATH = '/home/zairex/Code/cibr/analysis/signals/data/ica_spectra/'
LOAD_PATH = '/home/zairex/Code/cibr/analysis/signals/data/fica/'

print "load components"
with open(LOAD_PATH + 'restmed_components.p') as f:
    components = pickle.load(f)

def get_spectrum(component_data, start=0, end=None):
    source_stft = component_data.source_stft
    info = component_data.info
    freqs = component_data.freqs
    if end:
        end = int(end * info['sfreq'] / STEP_SIZE)
    else:
        end = source_stft.shape[1] - 1

    if start:
        start = int(start * info['sfreq'] / STEP_SIZE)

    y = np.mean(10 * np.log10(np.abs(source_stft[:, start:end])), axis=-1)
    x = freqs
    return x, y

print "Plotting averages"
med_average = np.zeros((len(components[0].freqs),))
rest_average = np.zeros((len(components[0].freqs),))

x, _ = get_spectrum(components[0])

for component in components:
    rest_average += get_spectrum(component, start=0, end=85)[1]
    med_average += get_spectrum(component, start=100)[1]

rest_average = rest_average / float(len(components))
med_average = med_average / float(len(components))

# correct baseline
difference = np.mean(med_average[0:5]) - np.mean(rest_average[0:5])
med_average = med_average - difference

f, ax = plt.subplots()
ax.plot(x, rest_average)
ax.plot(x, med_average)
ax.set(title='Spectra of meditation (green) and resting (blue) averaged over subjects',
       xlabel='Frequency', ylabel='Power Spectral Density (dB)')
plt.show()

print "Saving data for stats"
data = [['rest_peak', 'rest_average', 'med_peak', 'med_average']] 

for component in components:
    _, rest_spectrum = get_spectrum(component, start=0, end=85)
    _, med_spectrum = get_spectrum(component, start=100)

    # correct baseline
    difference = np.mean(med_spectrum[0:5]) - np.mean(rest_spectrum[0:5])
    med_spectrum = med_spectrum - difference

    data.append([
        np.max(rest_spectrum),
        np.mean(rest_spectrum[(x > 7) & (x < 13)]),
        np.max(med_spectrum),
        np.mean(med_spectrum[(x > 7) & (x < 13)]),
    ])
    
path = SAVE_PATH + 'restmed.csv'
print "Saving to: " + path
np.savetxt(path, np.array(data), fmt="%s", delimiter=',')
