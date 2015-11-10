# Joins multiple raw files to one target file
# Usage:
#     python join_parts.py /path/to/n/source/files /path/to/dest/file
import mne
import sys

def read_raw(filename):
    if filename.endswith('.fif'):
        raw = mne.io.Raw(filename, preload=True)
    else:
        # assume egi
        raw = mne.io.read_raw_egi(filename)
    return raw


def main(source_files, dest_file):
    print "Reading source files"
    raw_objects = [read_raw(source) for source in source_files]
    
    print "Concatenating.."
    raw_objects[0].append(raw_objects[1:])

    print "Saving.."
    raw_objects[0].save(dest_file, overwrite=True)


if __name__ == '__main__':
    cla = sys.argv
    main(cla[1:-1], cla[-1])

