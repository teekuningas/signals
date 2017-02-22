"""
Usage:
    python combine_meggie_csv.py list_of_filenames

    For example if in current folder you have files:
      AC_A0043_mc_cmb_raw_spectrum_1.txt
      AC_A0043_mc_cmb_raw_spectrum_2.txt
      AC_A0043_mc_cmb_raw_spectrum_3.txt
      AC_A0048_mc_cmb_raw_spectrum_1.txt
      AC_A0048_mc_cmb_raw_spectrum_2.txt
      AC_A0048_mc_cmb_raw_spectrum_3.txt

    To get all files in condition 1 you can either list them all:
      python combine_meggie_csv.py AC_A0043_mc_cmb_raw_spectrum_1.txt AC_A0048_mc_cmb_raw_spectrum_1.txt
    
    or you can use *-operator if you know how:
      python combine_meggie_csv.py *_1*

    or to get all .txt-files in current folder:
      python combine_meggie_csv.py *.txt

"""

import sys

contents = {}
filenames = sys.argv[1:]

for fname in filenames:
    with open(fname, 'rb') as f:
        contents[fname] = f.readlines()

# check freqs cohesion
freqs_valid = True
freqs_line = contents.values()[0][0]
for content in contents.values():
    if freqs_line != content[0]:
        valid = False
        break

if not freqs_valid:
    print "Some of the files have different freqs, don't know how to continue. :("
    exit(0)

print "Please give a list of channel numbers separated by spaces to select what channels to retain."
print "For EEG for example: 002 099 123"
print "For MEG for example: 0121 2121"
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
            filtered.append(sub_name + ' ' + line)

csv_lines = [freqs_line] + sorted(filtered)

combined_fname = raw_input("Please give a filename for new combined file: ")
with open(combined_fname, 'wb') as f:
    f.writelines(csv_lines)

