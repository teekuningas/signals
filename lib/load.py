import mne

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
    'KH005': {
        'med': ['MI_KH005_Meditaatio 201302.001',
                'MI_KH005_Meditaatio 201302.002'],
        'eoec': ['MI_EOEC_KH005 20130218 1711.raw']
    },
    'KH007': {
        'med': ['MI_KH007_Meditaatio 201302.001',
                'MI_KH007_Meditaatio 201302.002'],
        'eoec': ['MI_KH007_EOEC 20130220 0958.raw']
    },
    'KH011': {
        'med': ['MI_KH011_MEDITAATIO 201306.001',
                'MI_KH011_MEDITAATIO 201306.002'],
        'eoec': ['MI_KH011_EOEC 20130625 0936.raw']
    },
    'KH013': {
        'med': ['MI_KH13_meditaatio 2013062.001',
                'MI_KH13_meditaatio 2013062.002'],
        'eoec': ['MI_KH13_EOEC 20130626 1338.raw']
    },
    'KH014': {
        'med': ['MI_KH014_meditaatio 201306.001',
                'MI_KH014_meditaatio 201306.002'],
        'eoec': ['MI_KH014_EOEC 20130627 0947.raw']
    },
}


def _get_raw(filenames, clean_channels=False):
    parts = [mne.io.read_raw_egi(folder + fname, preload=True) for fname in filenames]
    raw = parts[0]
    raw.append(parts[1:])

    raw.filter(l_freq=1, h_freq=100)

    if clean_channels:
        picks = mne.pick_types(raw.info, eeg=True)
        raw.drop_channels([name for idx, name in enumerate(raw.info['ch_names'])
                           if idx not in picks])

    return raw


def get_raw(code, type_, clean_channels=False):
    return _get_raw(subjects[code][type_], clean_channels=clean_channels)

