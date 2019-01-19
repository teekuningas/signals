import os
import sys
import math

import mne
import numpy as np

triggers = dict([(i, 2**(i-1)) for i in range(1, 16 + 1)])

fname = sys.argv[2]
save_path = sys.argv[1]

total_amount_of_tasks = 48

raw = mne.io.read_raw_fif(fname, preload=True)

def get_events(id_):

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

    events = mne.find_events(raw, consecutive=True, shortest_event=1, uint_cast=True, verbose='warning')
    events = np.array(filter(lambda event: should_take(event), events))

    return events

def find_answer(type_, start, end):
    """
    """

    sfreq = raw.info['sfreq']

    button_11 = [event[0] for event in get_events(triggers[11])]
    button_12 = [event[0] for event in get_events(triggers[12])]
    question_marks = [event[0] for event in get_events(triggers[5])]

    question_mark_ind = np.where((question_marks > start + 2*sfreq) & (question_marks < end + 1*sfreq))[0]
    question_time = question_marks[question_mark_ind[0]]

    # add few samples to ensure button presses arent missed because they 
    # happen at the same time as start of new section
    button_11_ind = np.where((button_11 > question_time) & (button_11 < end + 1*sfreq))[0]
    button_12_ind = np.where((button_12 > question_time) & (button_12 < end + 1*sfreq))[0]

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

    if button_11_ind.size > 0:
        answer_time = button_11[button_11_ind[0]] - question_time
    elif button_12_ind.size > 0:
        answer_time = button_12[button_12_ind[0]] - question_time

    answer_time = answer_time / sfreq

    task = [None, None]
    if type_ == 15:
        task[0] = 'heart'
    elif type_ == 16:
        task[0] = 'note'

    if cond_1_ind.size > 0 and cond_2_ind.size > 0:
        task[1] = 'sync_with_tone'
    elif cond_1_ind.size > 0 and cond_2_ind.size == 0:
        task[1] = 'sync_without_tone'
    elif cond_3_ind.size > 0:
        task[1] = 'desync_with_tone'
    elif cond_2_ind.size > 0 and cond_1_ind.size == 0:
        task[1] = 'desync_without_tone'

    if type_ == 15:

        # 1 + 2 or 1
        if (cond_1_ind.size > 0 and cond_2_ind.size > 0) or cond_1_ind.size > 0:
            if button_11_ind.size > 0:
                return False, answer_time, task
            if button_12_ind.size > 0:
                return True, answer_time, task

        # 2 or 3
        if cond_2_ind.size > 0 or cond_3_ind.size > 0:
            if button_11_ind.size > 0:
                return True, answer_time, task
            if button_12_ind.size > 0:
                return False, answer_time, task

    if type_ == 16:

        # 1 + 2 or 3
        if (cond_1_ind.size > 0 and cond_2_ind.size > 0) or cond_3_ind.size > 0:
            if button_11_ind.size > 0:
                return False, answer_time, task
            if button_12_ind.size > 0:
                return True, answer_time, task

        # 1 or 2
        if cond_1_ind.size > 0 or cond_2_ind.size > 0:
            if button_11_ind.size > 0:
                return True, answer_time, task
            if button_12_ind.size > 0:
                return False, answer_time, task

    import pdb; pdb.set_trace()
    raise Exception("No answer found")

hearts = get_events(triggers[15])
notes = get_events(triggers[16])
rests = get_events(triggers[4])

def remove_extra_ones(things):
    things = list(things)
    i = len(things) - 1 
    while True:
        if things[i][0] < things[i-1][0] + raw.info['sfreq']:
            del things[i]
        i -= 1
        if i == 0:
            break
    return np.array(things)

hearts = remove_extra_ones(hearts)
notes = remove_extra_ones(notes)
rests = remove_extra_ones(rests)

print "Number of found hearts is " + str(len(hearts))
print "Number of found notes is " + str(len(notes))

if len(hearts) != total_amount_of_tasks/2:
    print "Warning: number of found hearts is different from normal"

if len(notes) != total_amount_of_tasks/2:
    print "Warning: number of found notes is different from normal"

def find_end(start):
    heart_end = np.inf
    for heart in hearts:
        if heart[0] > start + raw.info['sfreq']:
            heart_end = heart[0]
            break

    note_end = np.inf
    for note in notes:
        if note[0] > start + raw.info['sfreq']:
            note_end = note[0]
            break

    rest_end = np.inf
    for rest in rests:
        if rest[0] > start + raw.info['sfreq']:
            rest_end = rest[0]
            break

    return np.min([heart_end, note_end, rest_end])

heart_answers = []
note_answers = []
sfreq = raw.info['sfreq']

for idx, heart in enumerate(hearts):
    start = heart[0]
    end = find_end(start)
    answer, answer_time, task = find_answer(15, start, end)
    heart_answers.append((
        (start - raw.first_samp)/sfreq, 
        answer, 
        answer_time,
        task))

for idx, note in enumerate(notes):
    start = note[0]
    end = find_end(start)
    answer, answer_time, task = find_answer(16, start, end)
    note_answers.append((
        (start - raw.first_samp)/sfreq, 
        answer, 
        answer_time,
        task))

answers = heart_answers + note_answers
answers = sorted(answers, key=lambda x: x[0])

print "Answers are: "
for timepoint, answer, answer_time, task in answers:
    print str(answer), "on the task started at", str(timepoint), "s. (answer time: ", str(answer_time), ")"

trues = len([answer for answer in answers if answer[1] == True])
falses = total_amount_of_tasks - trues
print "Trues: " + str(trues)
print "Falses: " + str(falses)
print "True " + str(float(trues)*100 / (trues + falses)) + "% of time."

print "Answers for hearts: "
for timepoint, answer, answer_time, task in heart_answers:
    print str(answer), "on the heart task started at", str(timepoint), "s. (answer time: ", str(answer_time), ")"

trues = len([answer for answer in heart_answers if answer[1] == True])
falses = total_amount_of_tasks/2 - trues
print "Trues: " + str(trues)
print "Falses: " + str(falses)
print "True " + str(float(trues)*100 / (trues + falses)) + "% of time."

print "Answers for notes: "
for timepoint, answer, answer_time, task in note_answers:
    print str(answer), "on the note task started at", str(timepoint), "s. (answer time: ", str(answer_time), ")"

trues = len([answer for answer in note_answers if answer[1] == True])
falses = total_amount_of_tasks/2 - trues
print "Trues: " + str(trues)
print "Falses: " + str(falses)
print "True " + str(float(trues)*100 / (trues + falses)) + "% of time."

# stats!

subject_code = fname.split('/')[-1].split('.fif')[0]

stuff_to_write = [
    subject_code,
    str(len([answer for answer in heart_answers if answer[1] == True])),
    str(len([answer for answer in heart_answers if answer[1] == False])),
    str(len([answer for answer in note_answers if answer[1] == True])),
    str(len([answer for answer in note_answers if answer[1] == False])),
    str(len([answer for answer in answers if answer[1] == True])),
    str(len([answer for answer in answers if answer[1] == False])),
]

if not os.path.exists(save_path):
    os.makedirs(save_path)

with open(os.path.join(save_path, 'summary.csv'), 'a') as f:
    f.write(', '.join(stuff_to_write) + '\n')

with open(os.path.join(save_path, subject_code + '.csv'), 'wb') as f:
    f.write('Task start, Task type, Correct answer, Answer time\n')
    for timepoint, correct, answer_time, task in heart_answers:
        f.write(', '.join([str(timepoint), ' '.join(task), str(correct), 
                            str(answer_time)]) + '\n')
    for timepoint, correct, answer_time, task in note_answers:
        f.write(', '.join([str(timepoint), ' '.join(task), str(correct), 
                            str(answer_time)]) + '\n')

