import pickle

import mne
import matplotlib.pyplot as plt
import numpy as np

from lib.fourier_ica import FourierICA


BAND = [4, 25]
COMPONENTS = 8
RADIUS = 15
WSIZE = 4096

FILENAMES = [
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH001_MED-raw.fif', 'experienced'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH002_MED-raw.fif', 'experienced'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH003_MED-raw.fif', 'experienced'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH004_MED-raw.fif', 'experienced'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH005_MED-raw.fif', 'experienced'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH007_MED-raw.fif', 'experienced'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH009_MED-raw.fif', 'experienced'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH016_MED-raw.fif', 'experienced'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH017_MED-raw.fif', 'experienced'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH024_MED-raw.fif', 'experienced'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokemattomat/KH011_MED-raw.fif', 'novice'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokemattomat/KH013_MED-raw.fif', 'novice'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokemattomat/KH014_MED-raw.fif', 'novice'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokemattomat/KH015_MED-raw.fif', 'novice'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokemattomat/KH019_MED-raw.fif', 'novice'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokemattomat/KH021_MED-raw.fif', 'novice'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokemattomat/KH023_MED-raw.fif', 'novice'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokemattomat/KH025_MED-raw.fif', 'novice'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokemattomat/KH026_MED-raw.fif', 'novice'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokemattomat/KH028_MED-raw.fif', 'novice'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokemattomat/KH029_MED-raw.fif', 'novice'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokemattomat/KH030_MED-raw.fif', 'novice'),  # noqa
    ('/home/zairex/Code/cibr/data/graduprosessoidut/kokemattomat/KH031_MED-raw.fif', 'novice'),  # noqa
]


class ComponentData(object):
    """
    """
    def __init__(self, source_stft, sensor_stft, source_psd, freqs, info):
        self.source_stft = source_stft
        self.source_psd = source_psd
        self.sensor_stft = sensor_stft
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


def get_trial_data(fica, info):
    source_stft = fica.source_stft

    components = []
    for i in range(COMPONENTS):
        source_component = source_stft[i] 
        sensor_component = fica.component_in_sensor_space(i)
        source_psd = np.mean(np.abs(source_stft[i]), axis=-1)
        component_data = ComponentData(
            source_stft=source_component,
            sensor_stft=sensor_component,
            source_psd=source_psd,
            freqs=fica.freqs,
            info=info.copy(),
        )
        components.append(component_data)
    return TrialData(components=components)


def plot_components(components):

    layout_fname = 'gsn_129.lout'
    layout_path = '/home/zairex/Code/cibr/materials/'
    layout = mne.channels.read_layout(layout_fname, layout_path)

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
        tfr_ = mne.time_frequency.AverageTFR(info, np.abs(source_component), 
                                             times, component.freqs, 1)

        axes = fig_.add_subplot(len(components), 1, i + 1)
        tfr_.plot(picks=[0], axes=axes, show=False, mode='logratio')

    plt.show()


if __name__ == '__main__':

    mne.utils.set_log_level('ERROR')

    try:
        print "Trying to load processed data from file"
        subjects = pickle.load(open("data/.fica_epochs.p", "rb"))
        print "Loading succeeded!"
    except:
        print "Loading failed, processing.."
        subjects = []

    if not subjects:
        for fname, type_ in FILENAMES:
            raw = mne.io.Raw(fname, preload=True)
            epochs = get_epochs(raw)
            sfreq = raw.info['sfreq']
            ficas = filter(bool, [get_fica(epoch, sfreq) for epoch in epochs])

            if not ficas:
                continue

            trials = []
            for fica in ficas:
                trials.append(get_trial_data(fica, raw.info))
            
            subject = SubjectData(trials=trials, path=fname, type_=type_)
            subjects.append(subject)

        print "Start pickling data.."
        pickle.dump(subjects, open("data/.fica_epochs.p", "wb"))

    plot_components(subjects[0].trials[0].components)

