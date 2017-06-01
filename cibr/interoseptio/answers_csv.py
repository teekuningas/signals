import mne
import math
import numpy as np
import sys


triggers = dict([(i, 2**(i-1)) for i in range(1, 16 + 1)])
fname = sys.argv[-1]

raw = mne.io.read_raw_fif(fname, preload=True)

def get_events(id_):
    """ helper function to get events with certain value without caring 
    if non-related bits are on or off. works almost always.
    """

    id_bin = '{0:016b}'.format(id_)
    mask_bin = '{0:016b}'.format(id_).replace('1', 'j')
        .replace('0', '1').replace('j', '0')

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

def remove_extra_ones(things):
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

def find_answer(type_, start, end):
    """
    """

    button_11 = [event[0] for event in get_events(triggers[11])]
    button_12 = [event[0] for event in get_events(triggers[12])]
    # add few samples to ensure button presses arent missed because they 
    # happen at the same time as start of new section
    button_11_ind = np.where((button_11 > start + 2000) & (button_11 < end + 1000))[0]
    button_12_ind = np.where((button_12 > start + 2000) & (button_12 < end + 1000))[0]
    if any([
        button_11_ind.size > 0 and button_12_ind.size > 0,
        button_11_ind.size == 0 and button_12_ind.size == 0
    ]):
        import pdb; pdb.set_trace()
        button_11 = [event[0] for event in get_events(triggers[11])]
        button_12 = [event[0] for event in get_events(triggers[12])]
        raise Exception("Weird button patterns between " + str(start) + " and " + str(end))

    cond_1 = [event[0] for event in get_events(triggers[1])]
    cond_2 = [event[0] for event in get_events(triggers[2])]
    cond_3 = [event[0] for event in get_events(triggers[3])]
    cond_1_ind = np.where((cond_1 > start) & (cond_1 < end))[0]
    cond_2_ind = np.where((cond_2 > start) & (cond_2 < end))[0]
    cond_3_ind = np.where((cond_3 > start) & (cond_3 < end))[0]

    if type_ == 15:

        # 1 + 2 or 1
        if (cond_1_ind.size > 0 and cond_2_ind.size > 0) or cond_1_ind.size > 0:
            if button_11_ind.size > 0:
                return False
            if button_12_ind.size > 0:
                return True

        # 2 or 3
        if cond_2_ind.size > 0 or cond_3_ind.size > 0:
            if button_11_ind.size > 0:
                return True
            if button_12_ind.size > 0:
                return False

    if type_ == 16:

        # 1 + 2 or 3
        if (cond_1_ind.size > 0 and cond_2_ind.size > 0) or cond_3_ind.size > 0:
            if button_11_ind.size > 0:
                return False
            if button_12_ind.size > 0:
                return True

        # 1 or 2
        if cond_1_ind.size > 0 or cond_2_ind.size > 0:
            if button_11_ind.size > 0:
                return True
            if button_12_ind.size > 0:
                return False

    import pdb; pdb.set_trace()
    raise Exception("No answer found")

hearts = get_events(triggers[15])
notes = get_events(triggers[16])
rests = get_events(triggers[4])

hearts = remove_extra_ones(hearts)
notes = remove_extra_ones(notes)
rests = remove_extra_ones(rests)

print "Number of found hearts is " + str(len(hearts))
print "Number of found notes is " + str(len(notes))

if len(hearts) != 24:
    raise Exception("Something is really wrong")

if len(notes) != 24:
    raise Exception("Something is really wrong")

def find_end(start):
    heart_end = np.inf
    for heart in hearts:
        if heart[0] > start + 1000:
            heart_end = heart[0]
            break

    note_end = np.inf
    for note in notes:
        if note[0] > start + 1000:
            note_end = note[0]
            break

    rest_end = np.inf
    for rest in rests:
        if rest[0] > start + 1000:
            rest_end = rest[0]
            break

    return np.min([heart_end, note_end, rest_end])

heart_answers = []
note_answers = []
sfreq = raw.info['sfreq']

for idx, heart in enumerate(hearts):
    start = heart[0]
    end = find_end(start)
    heart_answers.append((start/sfreq, find_answer(15, start, end)))

for idx, note in enumerate(notes):
    start = note[0]
    end = find_end(start)
    note_answers.append((start/sfreq, find_answer(16, start, end)))

answers = heart_answers + note_answers
answers = sorted(answers, key=lambda x: x[0])

print "Answers are: "
for timepoint, answer in answers:
    print str(answer), "on the task started at", str(timepoint), "s."

trues = len([answer for answer in answers if answer[1] == True])
falses = 48 - trues
print "Trues: " + str(trues)
print "Falses: " + str(falses)
print "True " + str(float(trues)*100 / (trues + falses)) + "% of time."

print "Answers for hearts: "
for timepoint, answer in heart_answers:
    print str(answer), "on the task started at", str(timepoint), "s."

trues = len([answer for answer in heart_answers if answer[1] == True])
falses = 24 - trues
print "Trues: " + str(trues)
print "Falses: " + str(falses)
print "True " + str(float(trues)*100 / (trues + falses)) + "% of time."

stuff_to_write = [
    fname.split('/')[0].split('_')[-1],
    str(len([answer for answer in heart_answers if answer[1] == True])),
    str(len([answer for answer in heart_answers if answer[1] == False])),
    str(len([answer for answer in note_answers if answer[1] == True])),
    str(len([answer for answer in note_answers if answer[1] == False])),
    str(len([answer for answer in answers if answer[1] == True])),
    str(len([answer for answer in answers if answer[1] == False])),
]

with open('answers.csv', 'a') as f:
    f.write(', '.join(stuff_to_write) + '\n')
