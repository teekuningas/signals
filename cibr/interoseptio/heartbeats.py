import mne
import sys
import matplotlib.pyplot as plt
import numpy as np

raw = mne.io.Raw(sys.argv[-1], preload=True)

try:
    heartbeats = [event[0] for event in mne.find_events(raw, stim_channel='STI006')]
except:
    print "No distinct heartbeat channel.. using the collection channel"
    heartbeats = [event[0] for event in mne.find_events(raw, mask=65503, mask_type='not_and', uint_cast=True)]

if not heartbeats:
    print "No hearbeat events found.."
    exit(0)

length = len(heartbeats)
x = np.zeros((length-1,)) 
y = np.zeros((length-1,))

for i in range(length-1):
    x[i] = heartbeats[i+1] / raw.info['sfreq']
    y[i] = 60.0 / ((heartbeats[i+1] - heartbeats[i]) / raw.info['sfreq'])

# print "Removing outliers"
# count = 0
# for i in range(len(y))[::-1]:
#     if i == 0:
#         others = y[1:3]
#     elif i == len(y) - 1:
#         others = y[-4:-2]
#     else:
#         others = [y[i-1]] + [y[i+1]]
# 
#     if y[i] > 1.15*np.mean(others) or y[i] < 0.85*np.mean(others):
#         x = np.delete(x, [i])
#         y = np.delete(y, [i])
#         count += 1
# print "Deleted " + str(count) + " outliers."

plt.plot(x, y)
plt.gca().set_ylim([30, 150])
plt.show(block=True)

avg = 60.0 * length / ((heartbeats[-1] - heartbeats[0]) / raw.info['sfreq'])
print "Average ticks / minute: " + str(avg)
