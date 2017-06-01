import math
import sys

import mne

import numpy as np
import matplotlib.pyplot as plt


SAVE_FOLDER = 'output/heartbeats/'

fname = sys.argv[-1]

raw = mne.io.read_raw_fif(fname, preload=True)


def _get_events(id_):
    """ helper function to get events with certain value without caring
    if non-related bits are on or off. 
    """

    id_bin = '{0:016b}'.format(id_)
    mask_bin = '{0:016b}'.format(id_).replace('1', 'j').replace('0', '1').replace('j', '0')

    def should_take(event):
        """ check if event has same non-masked bits as id_
        """
        event_bin = '{0:016b}'.format(event[2])

        take_event = True
        for i in range(len(mask_bin)):

            if int(mask_bin[i]) == 1:
                continue

            if int(id_bin[i]) != int(event_bin[i]):
                take_event = False
                break
        return take_event

    events = mne.find_events(raw, consecutive=True, shortest_event=1,
                             uint_cast=True, verbose='warning')
    return np.array(filter(lambda event: should_take(event), events))

def _remove_extra_ones(things):
    """ helper to remove spurious condition events
    """
    things = list(things)
    i = len(things) - 1
    while True:
        if things[i][0] < things[i-1][0] + 1000:
            del things[i]
        i -= 1
        if i == 0:
            break
    return np.array(things)

def get_ibi(remove_outliers=False):
    try:
	heartbeats = [event[0] for event in mne.find_events(raw, stim_channel='STI006')]
    except:
	print "No distinct heartbeat channel.. using the collection channel"
	heartbeats = [event[0] for event in mne.find_events(raw, mask=65503, mask_type='not_and', uint_cast=True)]

    length = len(heartbeats)
    x = np.zeros((length-1,))
    y = np.zeros((length-1,))

    for i in range(length-1):
	x[i] = (heartbeats[i+1] - raw.first_samp) / raw.info['sfreq']
	y[i] = (heartbeats[i+1] - heartbeats[i]) / raw.info['sfreq']

    if remove_outliers:
	print "Removing outliers"
	count = 0
	for i in range(len(y))[::-1]:
	    if i == 0:
		others = y[1:3]
	    elif i == len(y) - 1:
		others = y[-4:-2]
	    else:
		others = [y[i-1]] + [y[i+1]]
	
	    if y[i] > 1.15*np.mean(others) or y[i] < 0.85*np.mean(others):
		x = np.delete(x, [i])
		y = np.delete(y, [i])
		count += 1
	print "Deleted " + str(count) + " outliers."

    return x, y


def intervals_from_condition(concentration, stimulus_type, correct=None):
    """
    concentration: heart / note
    stimulus_type: sync_with_tone / sync_without_tone /
                   desync_with_tone / desync_without_tone
    correct: True / False
    """

    # helper structure to get ids
    types = {
        'heart': 15,
        'note': 16,
    }

    # helper structure to get trigger id's from line numbers
    triggers = dict([(i, 2**(i-1)) for i in range(1, 16 + 1)])

    # get starting points for the intervals for given concentration task
    start_events = _get_events(triggers[types[concentration]])
    
    # composite channels are unfortunately ambigious
    # so make sure everything is fine
    start_events = _remove_extra_ones(start_events)

    # find ending points ands thus the intervals

    question_events = _get_events(triggers[5])

    def find_next(event):
        for question_event in question_events:
             # as a safety dont accept question events that are very close
             if question_event[0] > event[0] + 6*raw.info['sfreq']:
                 return question_event

    def is_correct_answer(interval, concentration, stimulus_type):

        start = interval[0]
        end = interval[1]

        # no answers
	button_11 = [event[0] for event in _get_events(triggers[11])]

        # yes answers
	button_12 = [event[0] for event in _get_events(triggers[12])]

	# add few samples to ensure button presses arent missed because they
	# happen at the same time as start of new section
	button_11_present = np.where((button_11 > start + 8*raw.info['sfreq']) & 
                                     (button_11 < end + 8*raw.info['sfreq']))[0].size > 0  # noqa
	button_12_present = np.where((button_12 > start + 8*raw.info['sfreq']) & 
                                     (button_12 < end + 8*raw.info['sfreq']))[0].size > 0  # noqa

        if button_11_present and button_12_present:
            print "Both buttons present for " + str(interval)
            return False

        if not button_11_present and not button_12_present:
            print "Neither of buttons present for " + str(interval)
            return False
         
        if button_11_present:
            if concentration == 'heart' and stimulus_type in ['desync_with_tone',
                                                              'desync_without_tone']:
                return True
            if concentration == 'note' and stimulus_type in ['sync_without_tone',
                                                              'desync_without_tone']:
                return True
        if button_12_present:
            if concentration == 'heart' and stimulus_type in ['sync_with_tone',
                                                              'sync_without_tone']:
                return True

            if concentration == 'note' and stimulus_type in ['sync_with_tone',
                                                              'desync_with_tone']:
                return True

        return False

    def select_types(interval, stimulus_type):

        start = interval[0]
        end = interval[1]

        trig_1 = [event[0] for event in _get_events(triggers[1])]
        trig_2 = [event[0] for event in _get_events(triggers[2])]
        trig_3 = [event[0] for event in _get_events(triggers[3])]

        trig_1_present = np.where((trig_1 > start - 2*raw.info['sfreq']) & (trig_1 < end))[0].size > 0
        trig_2_present = np.where((trig_2 > start - 2*raw.info['sfreq']) & (trig_2 < end))[0].size > 0
        trig_3_present = np.where((trig_3 > start - 2*raw.info['sfreq']) & (trig_3 < end))[0].size > 0

        if stimulus_type == 'sync_with_tone':
            if trig_1_present and trig_2_present:
                return True
        if stimulus_type == 'sync_without_tone':
            if trig_1_present and not trig_2_present:
                return True
        if stimulus_type == 'desync_with_tone':
            if trig_3_present:
                return True
        if stimulus_type == 'desync_without_tone':
            if trig_2_present and not trig_1_present:
                return True

        return False

    intervals = [(start[0], find_next(start)[0]) for start in start_events]

    intervals = filter(lambda ival: select_types(ival, stimulus_type),
                       intervals)

    if correct is True:
        intervals = filter(lambda ival: is_correct_answer(ival, 
                                                          concentration,
                                                          stimulus_type),
                           intervals)
    elif correct is False:
        intervals = filter(lambda ival: not is_correct_answer(ival, 
                                                              concentration, 
                                                              stimulus_type),
                           intervals)

    # convert to seconds
    intervals = [((start - raw.first_samp) / raw.info['sfreq'],
                  (end - raw.first_samp) / raw.info['sfreq']) 
                 for start, end in intervals]

    return intervals

def ibi_stats(ibi, intervals):
    indices_list = []
    for interval in intervals:
        indices_list.append(np.where((ibi[0] > interval[0]) & (ibi[0] < interval[1]))[0])

    indices = np.concatenate(indices_list)

    mean_ = np.mean(ibi[1][indices[:-1]])
    std_ = np.std(ibi[1][indices[:-1]])
    return mean_, std_
     

ibi = get_ibi(remove_outliers=True)

# Create a plot and save it to file

heart_intervals = sum([
    intervals_from_condition('heart', 'desync_without_tone'),
    intervals_from_condition('heart', 'sync_without_tone'),
    intervals_from_condition('heart', 'desync_with_tone'),
    intervals_from_condition('heart', 'sync_with_tone'),
], [])

note_intervals = sum([
    intervals_from_condition('note', 'desync_without_tone'),
    intervals_from_condition('note', 'sync_without_tone'),
    intervals_from_condition('note', 'desync_with_tone'),
    intervals_from_condition('note', 'sync_with_tone'),
], [])

fig = plt.figure()

fig.suptitle('IBI')

ax = fig.add_subplot(1, 1, 1)

for ival in heart_intervals:
    ax.axvspan(ival[0], ival[1], color='r', alpha=0.15, lw=2)

for ival in note_intervals:
    ax.axvspan(ival[0], ival[1], color='b', alpha=0.15, lw=2)

ax.set_xlabel('Time (s)')
ax.set_ylabel('IBI (ms)')

ax.plot(ibi[0], ibi[1])

plt.show()

try:
    os.makedirs(SAVE_FOLDER + 'pics/')
except:
    pass

save_path = SAVE_FOLDER + 'pics/' + raw.info['filename'].split('/')[-1].split('.')[0] + '.png'

print "Saving plot to " + save_path
fig.savefig(save_path)

data_array = []

print "Calculating IBI stats for different conditions"

for concentration in ['heart', 'note']:
    for stimulus_type in ['desync_without_tone', 'sync_without_tone', 'desync_with_tone', 'sync_with_tone']:
        for correct in [True, False]:
            intervals = intervals_from_condition(concentration, stimulus_type, correct)
            stats = ibi_stats(ibi, intervals)
            data_array.append([
                concentration + ' ' + stimulus_type + ' ' + str(correct),
                str(stats[0]),
                str(stats[1])
            ])

# add output to console
from pprint import pprint
pprint(data_array)

# and save to csv

try:
    os.makedirs(SAVE_FOLDER + 'data/')
except:
    pass

save_path = SAVE_FOLDER + 'data/' + raw.info['filename'].split('/')[-1].split('.')[0] + '.csv'
lines = [', '.join(row) + '\n' for row in [['concentration stimulus_type correct', 'mean', 'std']] + data_array]

with open(save_path, 'wb') as f:
    f.writelines(lines)
