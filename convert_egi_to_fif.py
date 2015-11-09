# Reads raw EGI and saves as FIF
# Usage: 
#     python convert_egi_to_fif.py /path/to/source/file /path/to/destination/file
import mne
import sys

def main(from_file, to_file):

    print "Read EGI from " + from_file
    raw = mne.io.read_raw_egi(from_file)

    print "Saving FIF to " + to_file
    raw.save(to_file)


if __name__ == '__main__':
    cla = sys.argv
    main(cla[1], cla[2])
