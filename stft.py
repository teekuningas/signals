import numpy as np
import matplotlib.pyplot as plt
import mne


def main():

    raw = mne.io.Raw('/home/zairex/Code/cibr/data/graduaineisto/meditaatio/KH004_MED-raw.fif', preload=True)
    # raw = mne.io.Raw('/home/zairex/Code/cibr/data/graduaineisto/EOEC/KH004_EOEC-raw.fif', preload=True)

    data = raw._data
    info = raw.info
    sfreq = info['sfreq']
    wsize = int(sfreq/2)
    tstep = int(wsize/2)
    channels = [
        16, # middle front 
        75, # middle back
        114, # middle right
        44, # middle left
    ]
    interval = 15 * sfreq

    events = mne.find_events(raw)

    tfr = mne.time_frequency.stft(data, wsize, tstep)

    x = np.arange(0, tfr.shape[2]*tstep, tstep) / sfreq
    y = mne.time_frequency.stftfreq(wsize, sfreq)

    # find index for frequency limit
    freq_limit = 20
    freq_idx = int(y.shape[0] / ((sfreq / 2) / freq_limit))

    # find time indices
    times = []
    for event in events:
        if event[0] - interval < 0:
            start = 0
        else:
            start = event[0] - interval 
        if event[0] + interval > data.shape[1] - 1:
            end = data.shape[1] - 1
        else:
            end = event[0] + interval
        
        start_index = int(start/tstep)
        end_index = int(end/tstep)

        times.append((start_index, end_index))


    for time_interval in times:

        fig, axarray = plt.subplots(2,2)
        for idx, channel in enumerate(channels):
            temp_x = x[time_interval[0]:time_interval[1]]
            temp_y = y[:freq_idx]
            temp_z = tfr[channel-1][:freq_idx, 
                                    time_interval[0]:time_interval[1]]

            ax = axarray[idx%2, idx/2]
            ax.pcolormesh(temp_x, temp_y, 10 * np.log10(temp_z), 
                          shading='gouraud')
            ax.axis('tight')

        plt.show()


if __name__ == '__main__':
    main()
