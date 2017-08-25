"""
Usage:
    python filter_and_combine.py output_file list_of_input_files

    Takes csv files (with similar header) as input and combines them
    to a new csv file with optional filtering (take only rows with
    specified substrings in the first column) and averaging
    (average all (filtered) rows to one row).

    Substrings (e.g channels) and averaging can be specified via 
    command line arguments or by interactive prompts.

    For example if in current folder you have files:
      AC_A0043_mc_cmb_raw_spectrum_1.csv
      AC_A0043_mc_cmb_raw_spectrum_2.csv
      AC_A0043_mc_cmb_raw_spectrum_3.csv
      AC_A0048_mc_cmb_raw_spectrum_1.csv
      AC_A0048_mc_cmb_raw_spectrum_2.csv
      AC_A0048_mc_cmb_raw_spectrum_3.csv

    To get all files in condition 1 you can either list them all:
      python filter_and_combine.py output.csv AC_A0043_mc_cmb_raw_spectrum_1.csv AC_A0048_mc_cmb_raw_spectrum_1.csv

    or you can use *-operator if you know how:
      python filter_and_combine.py output.csv *_1*

    or to get all .csv-files in current folder:
      python filter_and_combine.py output.csv *.csv

    Average over group (of subjects), and channels (specified in channels.txt):

      $ echo "0121 2121" > channels.txt
      $ python filter_and_combine.py --channels channels.txt --average group_average_0121_2121.csv subject_1.csv subject_2.csv subject_3.csv

"""

import sys
import os

import numpy as np


arguments = sys.argv

# are channels defined in a file?
CHANNELS_DEFINED = False
if '--channels' in arguments:
    idx = arguments.index('--channels')
    fname = arguments.pop(idx + 1)
    arguments.pop(idx)

    with open(fname, 'rb') as f:
        channels = f.read()

    CHANNELS_DEFINED = True

# Should average rows together?
AVERAGE_DEFINED = False
if '--average' in arguments:
    idx = arguments.index('--average')
    arguments.pop(idx)
    average = True

    AVERAGE_DEFINED = True

# Do some sanity checks

if len(arguments) <= 2:
    raise Exception("You must specify the output filename and at least one input file")

if os.path.exists(arguments[1]):
    raise Exception("First argument should be the output file and it should not exist.")

for path in arguments[2:]:
    if not os.path.exists(path):
        raise Exception(path + ' does not exist.')

contents = {}
filenames = arguments[2:]

for fname in filenames:
    with open(fname, 'rb') as f:
        contents[fname] = f.readlines()

# check header cohesion
header_valid = True
header_line = contents.values()[0][0]
for content in contents.values():
    if header_line != content[0]:
        valid = False
        break

if not header_valid:
    raise Exception("Some of the files have different headers, don't know what to do :(")

# remove header from all
for key in contents.keys():
    del contents[key][0]

if not CHANNELS_DEFINED:
    print "Please give a list of channel numbers separated by spaces to select what channels to retain."
    print "For EEG for example: 002 099 123"
    print "For MEG for example: 0121 2121"
    print "For all channels use empty string"
    print "Or more generally you can use any substrings that should be present in the first column."
    channels = raw_input('Write it here: ')

channels = [str(channel) for channel in channels.split(' ')]

filtered = []
for sub_name, content in contents.items():
    for line in content:
        found = False
        for channel in channels:
            if channel in line.split(',')[0]:
                found = True
                break
        if found:
            if len(contents) > 1:
                filtered.append(sub_name + ' ' + line)
            else:
                filtered.append(line)

if not AVERAGE_DEFINED:
    resp = raw_input('Do you want to average (y or n): ')
    average = True if resp == 'y' else False

if average:
    content_rows = np.array([np.array([float(val) for val in row.split(', ')[1:]])  # noqa
                             for row in filtered])
    averaged = np.mean(content_rows, axis=0)
    filtered = ['average, ' + ', '.join([str(val) for val in averaged.tolist()]) + '\n']

# save the file
csv_lines = [header_line] + sorted(filtered)

savepath = arguments[1].split('/')

if len(savepath) > 1:
    try:
        os.makedirs('/'.join(savepath[:-1]))
    except:
        pass

with open(arguments[1], 'wb') as f:
    f.writelines(csv_lines)

