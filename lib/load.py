import sys

import mne

raw_folder = '/home/zairex/Code/cibr/data/MI_eggie/'
processed_folder = ''

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
    'KH006': {
        'med': ['MI_meditaatio_KH006 201302.001',
                'MI_meditaatio_KH006 201302.002'],
        'eoec': ['MI_EOEC_2_KH006 20130219 1550.r']
    },
    'KH007': {
        'med': ['MI_KH007_Meditaatio 201302.001',
                'MI_KH007_Meditaatio 201302.002'],
        'eoec': ['MI_KH007_EOEC 20130220 0958.raw']
    },
    'KH008': {
        'med': ['MI_KH008_Meditaatio 201302.001',
                'MI_KH008_Meditaatio 201302.002'],
        'eoec': ['MI_KH008_EOEC 20130221 1037.raw']
    },
    'KH009': {
        'med': ['MI_KH009_meditaatio 201302.001',
                'MI_KH009_meditaatio 201302.002'],
        'eoec': ['MI_009_EOEC_oikea 20130227 1739']
    },
    'KH010': {
        'med': ['MI_KH010_meditation 201303.001',
                'MI_KH010_meditation 201303.002'],
        'eoec': ['MI_KH010_EOEC 20130313 1758.raw']
    },
    'KH011': {
        'med': ['MI_KH011_MEDITAATIO 201306.001',
                'MI_KH011_MEDITAATIO 201306.002'],
        'eoec': ['MI_KH011_EOEC 20130625 0936.raw']
    },
    'KH012': {
        'med': ['MI_KH012_MEDITAATIO2 20130.001',
                'MI_KH012_MEDITAATIO2 20130.002'],
        'eoec': ['MI_KH012_EOEC 20130625 1341.raw']
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
    'KH015': {
        'med': ['MI_KH015_meditaatio 201306.001',
                'MI_KH015_meditaatio 201306.002'],
        'eoec': ['MI_KH015_EOEC 20130627 1332.raw']
    },
    'KH016': {
        'med': ['MI_KH16_meditation 2013102.001',
                'MI_KH16_meditation 2013102.002'],
        'eoec': ['MI_KH16_EOEC 20131024 1001.raw']
    },
    'KH017': {
        'med': ['MI_KH17_meditaatio 2013102.001',
                'MI_KH17_meditaatio 2013102.002'],
        'eoec': ['MI_KH17_EOEC_oikea 20131024 140']
    },
    'KH018': {
        'med': ['MI_KH18_meditaatio 2013102.001',
                'MI_KH18_meditaatio 2013102.002'],
        'eoec': ['MI_KH18_EOEC 20131025 0953.raw']
    },
    'KH019': {
        'med': ['MI_KH19_meditaatio 2013111.001',
                'MI_KH19_meditaatio 2013111.002'],
        'eoec': ['MI_KH19_EOEC 20131118 0931.raw']
    },
    'KH020': {
        'med': ['MI_KH20_medtaatio 20131121.001',
                'MI_KH20_medtaatio 20131121.002'],
        'eoec': ['MI_KH20_EOEC 20131121 1135.raw']
    },
    'KH021': {
        'med': ['MI_KH21_meditaatio 2013112.001',
                'MI_KH21_meditaatio 2013112.002'],
        'eoec': ['MI_KH21_eoec 20131127 1018.raw']
    },
    'KH022': {
        'med': ['MI_KH22_meditaatio 2013112.001',
                'MI_KH22_meditaatio 2013112.002'],
        'eoec': ['MI_KH22_eoec 20131127 1357.raw']
    },
    'KH023': {
        'med': ['MI_KH23_meditaatio 2014011.001',
                'MI_KH23_meditaatio 2014011.002'],
        'eoec': ['MI_KH23_EOEC 20140113 1034.raw']
    },
    'KH024': {
        'med': ['MI_KH24_meditaatio 2014011.001',
                'MI_KH24_meditaatio 2014011.002'],
        'eoec': ['MI_KH24_EOEC 20140116 1707.raw']
    },
    'KH025': {
        'med': ['MI_KH25_meditaatio 2014012.001',
                'MI_KH25_meditaatio 2014012.002'],
        'eoec': ['MI_KH25_EOEC 20140122 1710.raw']
    },
    'KH026': {
        'med': ['MI_KH26_meditaatio 2014012.001',
                'MI_KH26_meditaatio 2014012.002'],
        'eoec': ['MI_KH26_EOEC 20140123 1503.raw']
    },
    'KH027': {
        'med': ['MI_KH27_meditaatio 2014012.001',
                'MI_KH27_meditaatio 2014012.002'],
        'eoec': ['MI_KH27_EOEC 20140128 0958.raw']
    },
    'KH028': {
        'med': ['MI_KH28_meditaatio 2014012.001',
                'MI_KH28_meditaatio 2014012.002'],
        'eoec': ['MI_KH28_EOEC 20140129 1745.raw']
    },
    'KH029': {
        'med': ['MI_KH29_meditaatio 2014021.001',
                'MI_KH29_meditaatio 2014021.002'],
        'eoec': ['MI_KH29_EOEC 20140211 1758.raw']
    },
    'KH030': {
        'med': ['MI_KH30_meditaatio 2014021.001',
                'MI_KH30_meditaatio 2014021.002'],
        'eoec': ['MI_kh30_EOEC 20140212 1721.raw']
    },
    'KH031': {
        'med': ['MI_KH31_meditaatio 2014021.001',
                'MI_KH31_meditaatio 2014021.002'],
        'eoec': ['MI_KH31_EOEC 20140214 1404.raw']
    },


}


def _get_raw(filenames):
    parts = [mne.io.read_raw_egi(raw_folder + fname, preload=True) 
             for fname in filenames]
    raw = parts[0]
    raw.append(parts[1:])

    raw.filter(l_freq=1, h_freq=100)

    return raw


def get_raw(code, type_):
    return _get_raw(subjects[code][type_])

def cli_raws():
    raws = []
    args = sys.argv
    for idx, arg in enumerate(args):
        if arg == '--rawkw':
            key_str = args[idx+1]
            code, type_ = key_str.split(':')
            raws.append(get_raw(code, type_))
        if arg == '--rawfif':
            path = args[idx+1]
            raws.append(mne.io.Raw(path, preload=True))
    return raws


def load_layout(MEG=False):
    layout_path = '/home/zairex/Code/cibr/materials/'
    if MEG:
        layout_filename = 'neuromag306all.lay'
    else:
        layout_filename = 'gsn_129.lout'
    layout = mne.channels.read_layout(layout_filename, layout_path)
    return layout
