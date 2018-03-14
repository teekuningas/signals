import numpy as np
import matplotlib.pyplot as plt


x = [1, 2, 2, 3, 2, 2, 1, 1, 1, 1, 1, 1]
y = [1, 1, 1, 1, 1, 1, 2, 2, 3, 2, 2, 1]

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(x, alpha=0.7)
ax1.plot(y, alpha=0.7)

corr = np.correlate(x, y, mode='full')
max_corr = np.argmax(corr)
displacement = max_corr - len(x) + 1
print "Displacement: ", str(displacement)

padded_x = np.pad(x, (max(0, displacement), -min(0, displacement)), 
                  mode='edge')
rolled_x = np.roll(padded_x, -displacement)
moved_x = rolled_x[max(0, displacement):len(rolled_x)+min(0, displacement)]

ax2.plot(moved_x, alpha=0.7)
ax2.plot(y, alpha=0.7)

plt.show()


