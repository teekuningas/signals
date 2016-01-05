# Divides one raw file to many raw files
# Usage:
#     python divide_to_subparts.py /path/to/source/file /destination/file/template_%-raw.fif
import mne
import sys
from copy import deepcopy

def read_raw(filename):
    if filename.endswith('.fif'):
        raw = mne.io.Raw(filename, preload=True)
    else:
        # assume egi
        raw = mne.io.read_raw_egi(filename)
    return raw


def sliding_crop(raw, interval):
    """ Slides data backwards and then crops to get a clean result """
    start = raw.time_as_index(interval[0])
    end = raw.time_as_index(interval[1])
    part = deepcopy(raw)
    part._data = raw._data[:, start:end]
    part = part.crop(tmin=0, tmax=interval[1]-interval[0]-1)
    return part


def main(source, template):
    raw = read_raw(source)
    raw.plot()

    intervals = []
    
    counter = 1
    while True:
        print str(counter) + ". part: "
        user_input = raw_input('Please give an interval in format start-end (empty input stops here): ')
        if not user_input:
            break
        intervals.append((float(user_input.split('-')[0]), float(user_input.split('-')[1]))) 
        counter += 1

    counter = 1
    for interval in intervals:
        part = sliding_crop(raw, interval)
        part.save(template.replace('%', str(counter)))
        counter += 1


if __name__ == '__main__':
    cla = sys.argv
    main(cla[1], cla[2])
