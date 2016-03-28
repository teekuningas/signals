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


def deviation(previous, current):
    result = 0
    for t in range(len(previous)):
        result += (previous[t] - current[t])**2 / previous[t]**2
    return result


data = raw._data
deviation_limit = 0.3
start = 10000
end = 11000
series = data[0][start:end]
x = range(0, end - start)

imfs = []
residue = series

while True:
    component = residue
    quit = False
    while True:
        try:
            max_env_func = extrap1d(envelope(component, argrelmax(component)[0]))
            min_env_func = extrap1d(envelope(component, argrelmin(component)[0]))
            max_env = max_env_func(x)
            min_env = min_env_func(x)
        except:
            quit = True
            break
        mean_env = (max_env + min_env) / 2

        last_component = component
        component = component - mean_env

        dev = deviation(last_component, component)
        if deviation_limit > dev:
            print "Imf created"
            break

    if quit:
        break

    imfs.append(component)
    residue = residue - component

    if np.all(np.diff(residue) >= 0):
        break

fig, axarray = plt.subplots(len(imfs) + 1)
for idx, ax in enumerate(axarray):
    if idx == 0:
        ax.plot(series)
    else:
        ax.plot(imfs[idx - 1])
plt.show()

print "miau"
