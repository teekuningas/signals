import mne
import numpy as np
from scipy.signal import argrelmax
from scipy.signal import argrelmin
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

raw = mne.io.Raw('/home/zairex/Code/cibr/data/gradudemo/KH005_MED-pre.fif', preload=True)

# http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range
def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))

    return ufunclike


def envelope(series, extrema_args):
    x = extrema_args
    y = series[extrema_args]
    return interp1d(x, y, kind='cubic')


def deviation(component1, component2):
    return 0.2

def is_monotonic(component):
    return True


data = raw._data
deviation_limit = 0.3
start = 0
end = 5000
series = data[0][start:end]
x = range(start, end)

imfs = []
residue = series

while True:
    component = residue
    while True:
        max_env_func = extrap1d(envelope(series, argrelmax(series)[0]))
        min_env_func = extrap1d(envelope(series, argrelmin(series)[0]))
        max_env = max_env_func(x)
        min_env = min_env_func(x)
        mean_env = (max_env + min_env) / 2
        last_component = component
        component = component - mean_env

        if deviation_limit > deviation(component, last_component):
            break

    imfs.append(component)
    residue = residue - component

    if is_monotonic(residue):
        break

# plt.plot(x, max_env)
# plt.plot(x, min_env)
# plt.plot(x, mean_env)
# plt.show()

print "miau"
