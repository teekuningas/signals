# Takes directory and find events counts for every file
# Usage:
#     python show_event_counts.py /path/to/source/directory
import mne
import sys
import glob

def read_raw(filename):
    if filename.endswith('.fif'):
        raw = mne.io.Raw(filename)
    else:
        # assume egi
        raw = mne.io.read_raw_egi(filename)
    return raw


def main(directory):
    search_path = directory + '/*.fif'
    filenames = glob.glob(search_path)
    event_counts = []
    corrupted = []
    for filename in filenames:
        try:
            raw = read_raw(filename)
            events = mne.find_events(raw)
            event_counts.append((filename.split('/')[-1], events.shape[0]))
        except:
            corrupted.append(filename.split('/')[-1])

    event_counts.sort(key=lambda tup: tup[0])

    print "Event counts: "
    for event_count in event_counts:
        print event_count

    print "More than 9 triggers"
    for event_count in event_counts:
        if event_count[1] > 9:
            print event_count

    print "Corrupted: "
    for corrupt in corrupted:
        print corrupt


if __name__ == '__main__':
    cla = sys.argv
    main(cla[1])
