import pickle
import sys

import mne
import matplotlib.pyplot as plt
import numpy as np

from lib.fourier_ica import FourierICA
from lib.cluster import cluster_components as cluster_matrix


BAND = [4, 25]
COMPONENTS = 8
RADIUS = 15
WSIZE = 4096

SOURCE_FOLDER = '/home/zairex/Code/cibr/data/graduprosessoidut'
RESULT_FOLDER = '/home/zairex/Code/cibr/analysis/data'

FILENAMES = [
    ('/kokeneet/KH001_MED-raw.fif', 'experienced'), 
    ('/kokeneet/KH002_MED-raw.fif', 'experienced'), 
    ('/kokeneet/KH003_MED-raw.fif', 'experienced'), 
    ('/kokeneet/KH004_MED-raw.fif', 'experienced'), 
    ('/kokeneet/KH005_MED-raw.fif', 'experienced'), 
    ('/kokeneet/KH007_MED-raw.fif', 'experienced'), 
    ('/kokeneet/KH009_MED-raw.fif', 'experienced'), 
    ('/kokeneet/KH016_MED-raw.fif', 'experienced'), 
    ('/kokeneet/KH017_MED-raw.fif', 'experienced'), 
    ('/kokeneet/KH024_MED-raw.fif', 'experienced'), 
    ('/kokemattomat/KH011_MED-raw.fif', 'novice'), 
    ('/kokemattomat/KH013_MED-raw.fif', 'novice'), 
    ('/kokemattomat/KH014_MED-raw.fif', 'novice'), 
    ('/kokemattomat/KH015_MED-raw.fif', 'novice'), 
    ('/kokemattomat/KH019_MED-raw.fif', 'novice'), 
    ('/kokemattomat/KH021_MED-raw.fif', 'novice'), 
    ('/kokemattomat/KH023_MED-raw.fif', 'novice'), 
    ('/kokemattomat/KH025_MED-raw.fif', 'novice'), 
    ('/kokemattomat/KH026_MED-raw.fif', 'novice'), 
    ('/kokemattomat/KH028_MED-raw.fif', 'novice'), 
    ('/kokemattomat/KH029_MED-raw.fif', 'novice'), 
    ('/kokemattomat/KH030_MED-raw.fif', 'novice'), 
    ('/kokemattomat/KH031_MED-raw.fif', 'novice'), 
]


class ComponentData(object):
    """
    """
    def __init__(self, source_stft, sensor_stft, sensor_topo, source_psd, freqs, info):
        self.source_stft = source_stft
        self.sensor_stft = sensor_stft
        self.source_psd = source_psd
        self.sensor_topo = sensor_topo
        self.freqs = freqs
        self.info = info


class TrialData(object):
    """
    """
    def __init__(self, components):
        self.components = components


class SubjectData(object):
    """
    """
    def __init__(self, trials, path, type_):
        self.trials = trials
        self.path = path
        self.type = type_


def _filter_triggers(triggers, sfreq, start, end):
    """
    check if this trigger is ok with following conditions:
      * it is at least ``rad`` seconds away from other triggers
      * it is at least ``rad`` seconds away from start or end
    """

    rad = RADIUS * sfreq

    valid_triggers = []
    for trigger in triggers:

        if trigger < start + rad:
            continue

        if trigger > end - rad:
            continue

        valid = True

        for other in triggers:
            # don't compare to itself
            if other == trigger:
                continue

            if trigger < other + rad and trigger > other - rad:
                valid = False

        if not valid:
            continue

        valid_triggers.append(trigger)

    return np.array(valid_triggers, dtype=triggers.dtype)


def get_epochs(raw):
    sfreq = raw.info['sfreq']
    rad = RADIUS*sfreq

    # find and filter triggers
    triggers = mne.find_events(raw)[:, 0]
    triggers = _filter_triggers(triggers, sfreq, raw.first_samp, raw.last_samp)

    # drop bad and non-data channels
    raw.drop_channels(raw.info['ch_names'][128:] + raw.info['bads'])

    data = raw._data

    # create epochs
    epochs = []
    for trigger in triggers:
        epochs.append(data[:, int(trigger-rad):int(trigger+rad)])

    print str(len(epochs)) + " epochs found!"

    return epochs


def get_fica(epoch, sfreq):
    """
    """
    
    # calculate fourier ica
    try:
        fica = FourierICA(wsize=WSIZE, n_components=COMPONENTS, 
                          sfreq=sfreq, hpass=BAND[0], lpass=BAND[1])
        fica.fit(epoch)
        return fica
    except:
        # did not converge
        pass


def create_image(topo, layout, info, size=32):
    indices = [idx for idx, name in enumerate(layout.names)
               if name in info['ch_names']]
    image = np.zeros((size, size))
    for i in range(len(topo)):
        location = layout.pos[layout.names.index(info['ch_names'][i])]
        x = int(location[0] * 32)
        y = int(location[1] * 32)
        image[x, y] += topo[i]

    return image


def create_topo(sensor_stft, layout, info):

    # calculate "images"
    topo = np.sum(np.abs(sensor_stft), axis=(1, 2))

    # normalize
    topo = topo / np.linalg.norm(topo)

    # create comparable images
    image = create_image(topo, layout, info)

    return image


def get_trial_data(fica, info, layout):
    source_stft = fica.source_stft

    components = []
    for i in range(COMPONENTS):
        source_component = source_stft[i] 
        sensor_stft = fica.component_in_sensor_space(i)
        sensor_topo = create_topo(sensor_stft, layout, info)
        source_psd = np.mean(np.abs(source_stft[i]), axis=-1)
        component_data = ComponentData(
            source_stft=source_component,
            sensor_stft=sensor_stft,
            sensor_topo=sensor_topo,
            source_psd=source_psd,
            freqs=fica.freqs,
            info=info.copy(),
        )
        components.append(component_data)
    return TrialData(components=components)


def plot_components(components, layout):
    # create figure for head topographies
    fig_ = plt.figure()
    for i, component in enumerate(components):
        sensor_component = np.abs(component.sensor_stft)
        tfr_ = mne.time_frequency.AverageTFR(component.info, sensor_component, 
            range(sensor_component.shape[2]), component.freqs, 1)

        axes = fig_.add_subplot(len(components), 1, i + 1)
        mne.viz.plot_tfr_topomap(tfr_, layout=layout, axes=axes, show=False)

    # create figure ica components
    fig_ = plt.figure()
    for i, component in enumerate(components):

        # mock info
        info = component.info.copy()
        info['chs'] = [info['chs'][0]]

        rad = RADIUS
        source_component = (component.source_stft)[np.newaxis, :]

        times = np.arange(-rad, rad, float(rad*2)/source_component.shape[2])
        times = times / info['sfreq']
        tfr_ = mne.time_frequency.AverageTFR(info, np.abs(source_component), 
                                             times, component.freqs, 1)

        axes = fig_.add_subplot(len(components), 1, i + 1)
        tfr_.plot(picks=[0], axes=axes, show=False, mode='logratio')

    plt.show()


def cluster_components(subjects):

    # first create a flattened matrix out of subjects' component hierarchy
    component_matrix = []
    for subject in subjects:
        for trial in subject.trials:
            component_matrix.append(trial.components)

    # do the actual clustering
    ordered_matrix = cluster_matrix(component_matrix)

    # recreate the structured component hierarchy
    clustered = []
    idx = 0
    for subject in subjects:
        new_trials = []
        for trial in subject.trials:
            new_components = ordered_matrix[idx]
            new_trials.append(TrialData(
                components = new_components
            ))
            idx += 1

        clustered.append(SubjectData(
            trials=new_trials,
            path=subject.path,
            type_=subject.type
        ))

    return clustered


def main():

    mne.utils.set_log_level('ERROR')

    try:
        result_arg = filter(lambda x: 'RESULT=' in x, sys.argv)[0].split('=')[1]
    except:
        result_arg = RESULT_FOLDER

    try:
        source_arg = filter(lambda x: 'SOURCE=' in x, sys.argv)[0].split('=')[1]
    except:
        source_arg = SOURCE_FOLDER

    layout_fname = 'gsn_129.lout'
    layout_path = source_arg
    layout = mne.channels.read_layout(layout_fname, layout_path)

    input_ = raw_input('Load raw files, structured data or clustered data (r, s, c)? ')

    if input_ not in ['r', 's', 'c']:
        print "Quitting."
        return

    if input_ == 'r':
        print "Reading and processing data from files.."
        subjects = []

        for fname, type_ in FILENAMES:
            fname = source_arg + fname
            raw = mne.io.Raw(fname, preload=True)
            epochs = get_epochs(raw)
            sfreq = raw.info['sfreq']
            ficas = filter(bool, [get_fica(epoch, sfreq) for epoch in epochs])

            if not ficas:
                continue

            trials = []
            for fica in ficas:
                trials.append(get_trial_data(fica, raw.info, layout))
            
            subject = SubjectData(trials=trials, path=fname, type_=type_)
            subjects.append(subject)

        print "Start pickling data.."
        pickle.dump(subjects, open(result_arg + "/.fica_epochs.p", "wb"))

    if input_ == 's':
        print "Trying to load structured data from pickle file"
        subjects = pickle.load(open(result_arg + "/.fica_epochs.p", "rb"))
        print "Loading succeeded!"

    if input_ == 'c':
        print "Trying to load clustered data from pickle file"
        subjects = pickle.load(open(result_arg + "/.fica_epochs_clustered.p", "rb"))
        print "Loading succeeded!"
    else:
        subjects = cluster_components(subjects)

        print "Start pickling data.."
        pickle.dump(subjects, open(result_arg + "/.fica_epochs_clustered.p", "wb"))

    import pdb; pdb.set_trace()

    # plot_components(subjects[0].trials[0].components, layout)

    for i in range(COMPONENTS):
        int_components = []
        for subject in subjects:
            int_components.append(subject.trials[0].components[i])
        plot_components(int_components, layout)


if __name__ == '__main__':
    main()
