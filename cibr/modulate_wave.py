'''
Creates stereo sound from wave that has each channel modulated with different frequency

Usage:
python modulate_wave.py sound.wav

@author: erpipehe
'''
from scipy.io import wavfile
from scipy import signal
import numpy as np
import sys

depth = 0.95
frequencies = [38, 42]
butter_order = 7

filename = sys.argv[1] 
sample_rate, data = wavfile.read(filename)
sample_count = data.shape[0]

# halve amplitude to avoid overflows
data = data.astype(np.float16) / 2

# low-pass filter below 4khz
l_cutoff = 4000.0 / (sample_rate / 2)
b, a = signal.butter(butter_order, l_cutoff, btype='low')
data = np.swapaxes(signal.filtfilt(b, a, np.swapaxes(data, 0, 1)), 0, 1)

# add white noise 20db weaker than the speech
amplitude = np.max(data)*0.01
noise = np.zeros(sample_count, dtype=np.float16)
for i in range(len(noise)):
    noise[i] = 2 * amplitude * (np.random.random() - 0.5)
data = data + np.swapaxes(np.array((noise, noise)), 0, 1)

# create modulation waves
modulation_waves = np.zeros((len(frequencies), sample_count), dtype=np.float16)
for i in range(len(frequencies)):
    for j in range(sample_count):
        t = float(j)/sample_rate
        modulation_waves[i][j] = (1 + depth*np.sin(2*np.pi*frequencies[i]*t))  # noqa

# create first output wave
first_data = np.multiply(data, np.swapaxes(modulation_waves, 0, 1))

# swap channels for second output wave
second_data = first_data[:, ::-1]

# without_ext = '.'.join(filename.split('.')[:-1])
without_ext = filename.split('.')[-2].split('/')[-1]

# save first file
first_filename = without_ext + '_' + str(frequencies[0]) + 'left' + str(frequencies[1]) + 'right.wav'  # noqa
wavfile.write(first_filename, sample_rate, np.round(first_data).astype(np.int16))  # noqa

# save second file
second_filename = without_ext + '_' + str(frequencies[1]) + 'left' + str(frequencies[0]) + 'right.wav'  # noqa
wavfile.write(second_filename, sample_rate, np.round(second_data).astype(np.int16))  # noqa
