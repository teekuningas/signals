import numpy as np
import matplotlib.pyplot as plt
import mne


def main():
    raw = mne.io.Raw('/home/zairex/Code/cibr/demo/MI_KH009_MED-bads-raw-pre.fif', preload=True)
    # raw = mne.io.Raw('/home/zairex/Code/cibr/data/graduaineisto/EOEC/KH004_EOEC-raw.fif', preload=True)

    data = raw._data[:128]
    info = raw.info
    sfreq = info['sfreq']
    wsize = int(sfreq/2)
    tstep = int(wsize/2)
    channels = [
        11, # middle front 
        75, # middle back
        108, # middle right
        45, # middle left
    ]
    interval = 15 * sfreq

    events = mne.find_events(raw)

    tfr = mne.time_frequency.stft(data, wsize, tstep)

    y = mne.time_frequency.stftfreq(wsize, sfreq)

    # find index for frequency limit
    freq_limit = 20
    freq_idx = int(y.shape[0] / ((sfreq / 2) / freq_limit))

    # find time indices
    times = []
    for event in events:
        if event[0] - interval < 0:
            continue
        else:
            start = event[0] - interval 
        if event[0] + interval > data.shape[1] - 1:
            continue
        else:
            end = event[0] + interval
        
        start_index = int(start/tstep)
        end_index = int(end/tstep)

        times.append((start_index, end_index))

    averaged = tfr[:, :, times[0][0]:times[0][1]]
    for idx in range(len(times)-1):
        averaged += tfr[:, :, times[idx+1][0]:times[idx+1][1]]
    averaged = averaged / len(times)

    x = np.arange(0, averaged.shape[2]*tstep, tstep) / sfreq

    fig, axarray = plt.subplots(2,2)
    for idx, channel in enumerate(channels):
        temp_x = x[:]
        temp_y = y[:freq_idx]
        temp_z = averaged[channel-1][:freq_idx, :]

        ax = axarray[idx%2, idx/2]
        ax.set_title(str(channel))
        ax.pcolormesh(temp_x, temp_y, 10 * np.log10(temp_z), 
                      shading='gouraud')
        ax.axis('tight')

    plt.show()


if __name__ == '__main__':
    main()
