from collections import OrderedDict

import numpy as np


def extract_intervals_fdmsa_ic(events, sfreq, first_samp):
    """ meditation intervals """
    intervals = OrderedDict()

    trigger_info = [
        ('heart', 15),
        ('note', 16),
    ]

    for name, event_id in trigger_info:
        intervals[name] = []

    counter = 0
    for idx, event in enumerate(events):
        for name, bit in trigger_info:
            if int(format(event[2], '#020b')[-bit]) == 1:
                print(
                    str(format(event[2], '#020b')) + ', ' +
                    str(bit) + ', ' +
                    str(event))
                ival_start = event[0] + 1*sfreq
                ival_end = ival_start + 15*sfreq

                intervals[name].append((
                    (ival_start - first_samp) / sfreq,
                    (ival_end - first_samp) / sfreq))
                counter += 1
    if counter != 16: 
        print("Warning!!! " + str(counter) + " events found.")

    return intervals


def extract_intervals_multimodal(events, sfreq, first_samp):

    categories = {
        4: 'Visual lower left',
        2: 'Visual lower right',
        1: 'Visual upper right',
        8: 'Visual upper left',
        3: 'Auditory right',
        5: 'Auditory left',
        16: 'Somato left',
        32: 'Somato right'
    }
    intervals = OrderedDict()
    for event in events:
        if categories[event[2]] not in intervals:
            intervals[categories[event[2]]] = []
        intervals[categories[event[2]]].append(((event[0] - first_samp) / sfreq,
                                         (event[0] - first_samp) / sfreq + 1))
    return intervals

def extract_intervals_hengitys(events, sfreq, first_samp):

    categories = {
        1: 'Updown',
        2: 'Downup',
    }
    intervals = OrderedDict()
    for event in events:
        if categories[event[2]] not in intervals:
            intervals[categories[event[2]]] = []
        intervals[categories[event[2]]].append(((event[0] - first_samp) / sfreq,
                                         (event[0] - first_samp) / sfreq + 4))
    return intervals


def extract_intervals_fdmsa_rest(events, sfreq, first_samp, length):

    intervals = OrderedDict()

    intervals['EO'] = [(2*length / 3 + 5, length - 5)]
    intervals['EC'] = [(5, 2*length / 3 - 5)]
    return intervals


def extract_intervals_meditaatio(events, sfreq, first_samp, tasks):
    """ meditation intervals """
    intervals = OrderedDict()

    trigger_info = [
        ('mind', 10),
        ('rest', 11),
        ('plan', 12),
        ('anx', 13),
    ]

    for idx, event in enumerate(events):
        for name, event_id in trigger_info:
            if name not in tasks:
                continue
            if name not in intervals:
                intervals[name] = []
            if event[2] == event_id:
                ival_start = event[0] + 2*sfreq
                try:
                    ival_end = events[idx+1][0] - 2*sfreq
                except:
                    # last trigger (rest)
                    ival_end = event[0] + 120*sfreq - 2*sfreq

                intervals[name].append((
                    (ival_start - first_samp) / sfreq,
                    (ival_end - first_samp) / sfreq))

    return intervals


