import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

num_points = 30
powerlaw = lambda x, amp, index: amp * (x**index)

x = np.linspace(1.0, 20, num_points)

y = powerlaw(x, 10.0, -1.5)
y += np.random.randn(num_points) * 0.4 * y

# first do log-log transform
x_log = np.log10(x)
y_log = np.log10(y)

from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(x_log,y_log)

x_log_fitted = x_log.copy()
y_log_fitted = intercept + slope*x_log_fitted

x_fitted = 10**x_log_fitted
y_fitted = 10**y_log_fitted

fig = plt.figure()
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)

ax1.plot(x, y)
ax1.plot(x_fitted, y_fitted)
ax2.plot(x_log_fitted, y_log_fitted)
plt.show()

