import mne
import numpy as np

from scikits.audiolab import wavread


def find_peaks(data, threshold, gap_length):
    """ finds first value over threshold if there exists
    a valley in both sides """

    # rectify to make finding peaks easier
    thresholded_data = np.zeros(data.shape, dtype=np.int8)
    for x in range(data.shape[0]):
        if data[x] > threshold:
            thresholded_data[x] = 1
        else:
            thresholded_data[x] = 0

    filled_data = thresholded_data.copy()

    # fill small gaps
    for x in range(thresholded_data.shape[0]):
        if x % 10000 == 0:
            print str(float(x)/thresholded_data.shape[0]) + " %"
        sum_left = 0
        sum_right = 0
        for i in range(-gap_length, 0):
            if x + i >= 0:
                sum_left += thresholded_data[x + i]

        for i in range(1, gap_length + 1):
            if x + i < thresholded_data.shape[0]:
                sum_right += thresholded_data[x + i]

        if sum_left > 0 and sum_right > 0:
            filled_data[x] = 1
    
    # define peaks as the first 1 sample after 0's
    peaks = []
    for x in range(filled_data.shape[0] - 1):
        if filled_data[x] == 0 and filled_data[x+1] == 1:
            peaks.append(x+1)

    return np.array(peaks)


if __name__ == '__main__':

    audio_data, sfreq, encoding = wavread('delays.wav')

    # crop and decimate
    factor = 5
    interesting_area = [51000, 1250000]
    sfreq = float(sfreq) / factor
    audio_data = audio_data[interesting_area[0]:interesting_area[1]:factor]

    with open('logfile.txt', 'rb') as stim_file:
        stim_data = stim_file.readlines()

    stim_data = np.array([float(line.split(', ')[4]) for line in stim_data])

    # params
    peak_threshold = 0.22
    gap_length = int(0.25*sfreq)

    audio_peaks = find_peaks(audio_data, peak_threshold, gap_length) / float(sfreq)

    delays = np.abs(stim_data-audio_peaks)

    print "Std: " + str(np.std(delays))
    print "Max deviation: " + str(max(delays - np.mean(delays)))
