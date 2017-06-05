import sys
import os
import numpy as np

paths = sys.argv[1:]

# read into list of lists (rows of csvs)
subjects = []
for path in paths:
    with open(path, 'rb') as f:
        subjects.append(f.readlines())

# split into list of lists of lists
for i in range(len(subjects)):
    subjects[i] = [[term.strip('\n') for term in row.split(', ')] for row in subjects[i]]

TEMPLATE = {
    'heart_focus': [
        'heart desync_without_tone True',
        'heart desync_without_tone False',
        'heart sync_without_tone True',
        'heart sync_without_tone False',
        'heart desync_with_tone True',
        'heart desync_with_tone False',
        'heart sync_with_tone True',
        'heart sync_with_tone False',
    ],
    'note_focus': [
        'note desync_without_tone True',
        'note desync_without_tone False',
        'note sync_without_tone True',
        'note sync_without_tone False',
        'note desync_with_tone True',
        'note desync_with_tone False',
        'note sync_with_tone True',
        'note sync_with_tone False',
    ]
}

new_subjects = []
for i in range(len(subjects)):
    new_rows = []
    for new_key, conditions in TEMPLATE.items():
        
        # first calculate the combined average as a weighted average of within-group averages
        grand_mean, count = 0, 0
        for row in subjects[i]:
            if row[0] in conditions:
                if row[3] != '0':
                   grand_mean += float(row[1]) * float(row[3])
                   count += float(row[3])

        grand_mean = grand_mean / float(count)

        # then calculate the combined standard deviation basically as sqrt of GV = (ESS + TGSS) / (N-1)
        # where ESS and TGSS are familiar quantities from anova

        ESS = 0
        for row in subjects[i]:
            if row[0] in conditions:
                if row[3] != '0':
                    ESS += float(row[2])**2 * (float(row[3]) - 1)

        TGSS = 0
        for row in subjects[i]:
            if row[0] in conditions:
                if row[3] != '0':
                    TGSS += (float(row[1]) - grand_mean)**2 * float(row[3])

        GV = (ESS + TGSS) / (count - 1)

        grand_std = np.sqrt(GV)

        new_rows.append([new_key, grand_mean, grand_std, int(count)])
    new_subjects.append(new_rows)

# prepare for saving
for i in range(len(subjects)):
    name = paths[i].split('/')[-1][3:8]
    new_subjects[i] = [name + ' ' + ', '.join([str(val) for val in row]) + '\n' for row in new_subjects[i]]

# flatten
new_subjects = sum(new_subjects, [])

path = 'output/interoseptio/combined/'
try:
    os.makedirs(path)    
except Exception as e:
    print str(e)

with open(path + 'combined.csv', 'wb') as f:
    f.writelines(new_subjects)

import pdb; pdb.set_trace()
print "kissa"
