import numpy as np
from scipy.special import fresnel
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x, a, b1, b2):
    S, C = fresnel(x / a)
    y = np.array([C + b1, S + b2])
    return y.ravel()


xdata = np.array([1, 2, 3, 4], dtype=float)
ydata = func(xdata, 5, 0, 0).reshape(-1, len(xdata)).T
print(ydata)

popt, pcov = curve_fit(func, xdata, ydata.T.ravel())
print(popt, pcov)
y = func(xdata, *popt).reshape(-1, len(xdata)).T
print(y)
