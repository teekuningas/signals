import mne
import numpy as np
import matplotlib.pyplot as plt

folder = '/home/zairex/Code/cibr/data/MI_eggie/'

subjects = {
    'KH001': {
        'med': ['MI_KH001_Meditaatio 201302.001',
                'MI_KH001_Meditaatio 201302.002'],
        'eoec': ['MI_KH001_EOEC 20130211 0959.raw']
    },
    'KH002': {
        'med': ['MI_KH002_medidaatio 201302.001',
                'MI_KH002_medidaatio 201302.002'],
        'eoec': ['MI_KH002_EOEC 20130211 1704.raw']
    },
    'KH003': {
        'med': ['MI_KH003_meditaatio_2 2013.001',
                'MI_KH003_meditaatio_2 2013.002'],
        'eoec': ['MI_KH003_EOEC_OIKEA 20130213 15']
    },
    'KH004': {
        'med': ['MI_KH004_meditaatio2 20130.001',
                'MI_KH004_meditaatio2 20130.002'],
        'eoec': ['MI_KH004_EOEC 20130215 1641.raw']
    },

    'KH007': {
        'med': ['MI_KH007_Meditaatio 201302.001',
                'MI_KH007_Meditaatio 201302.002'],
        'eoec': ['MI_KH007_EOEC 20130220 0958.raw']
    },
}

# front, back, right, left
channels = [11, 75, 108, 45]

subject = subjects['KH003']
threshold = 0.01

# log, power?

layout_path = '/home/zairex/Code/cibr/materials/'
layout_filename = 'gsn_129.lout'
layout = mne.channels.read_layout(layout_filename, layout_path)

def get_raw(filenames):
    parts = [mne.io.read_raw_egi(folder + fname, preload=True) for fname in filenames]
    raw = parts[0]
    raw.append(parts[1:])

    raw.filter(l_freq=1, h_freq=100)

    picks = mne.pick_types(raw.info, eeg=True)
    raw.drop_channels([name for idx, name in enumerate(raw.info['ch_names']) 
                       if idx not in picks])

    return raw

eoec_raw = get_raw(subject['eoec'])
med_raw = get_raw(subject['med'])

raw = eoec_raw
raw.append(med_raw)

wsize = 4096
data = raw._data

stft = np.log(1 + np.abs(mne.time_frequency.stft(data, wsize)))
freqs = mne.time_frequency.stftfreq(wsize, raw.info['sfreq'])

# standardize
max_ = np.max(stft)
min_ = np.min(stft)
stft = (stft - min_) / (max_ - min_)

# clip
stft = np.clip(stft, 0, threshold)

stft = stft - threshold/2

tfr = mne.time_frequency.AverageTFR(
    raw.info, stft, np.arange(stft.shape[2], dtype=np.float64), freqs, 1
)

picks = mne.pick_types(raw.info, eeg=True)

for channel in channels:
    tfr.plot(picks=[channel], title=str(channel), fmin=1, fmax=100, show=False)

plt.show()

