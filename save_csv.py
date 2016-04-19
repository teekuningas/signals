# Authors: Erkka Heinila <erkka.heinila@jyu.fi>
#
# License: BSD (3-clause)

import sys

import numpy as np
import mne

def group_save_evokeds(filename, evokeds, names):
    """ Combine data from multiple evokeds to one big csv """

    if len(evokeds) == 0:
        raise Value("At least one evoked object is needed.")

    print "Writing " + str(len(evokeds)) + " subject's evoked data to csv."

    # gather all the data to list of rows
    all_data = []

    # time point data, assume same lengths for all evokeds
    all_data.append(['times'] + evokeds[0].times.tolist())

    # time series data
    for sub_idx, evoked in enumerate(evokeds):
        for ch_idx in range(len(evoked.data)):
            ch_name = evoked.info['ch_names'][ch_idx].replace(' ', '')
            row_name = names[sub_idx] + ' ' + ch_name

            # mark bad channels
            if evoked.info['ch_names'][ch_idx] in evoked.info['bads']:
                row_name += ' (bad)'

            row = [row_name] + evoked.data[ch_idx, :].tolist()
            all_data.append(row)

    # save to file
    all_data = np.array(all_data)
    np.savetxt(filename, all_data, fmt='%s', delimiter=', ')


if __name__ == '__main__':
    paths = sys.argv[1:]
    raws = [mne.io.Raw(path, preload=True) for path in paths]

    raws[0].info['bads'] = ['EEG 021']

    evokeds = []
    for raw in raws:
        events = mne.find_events(raw)
        epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.2)
        evoked = epochs.average()
        evokeds.append(evoked)

    names = ['.'.join(path.split('/')[-1].split('.')[:-1]) for path in paths]

    group_save_evokeds('test.csv', evokeds, names)
