from scipy.interpolate import interp1d
import numpy as np


def color_gradient(colors, vmin=0, vmax=1):
    colors = np.array(colors)
    interp = interp1d(np.linspace(vmin, vmax, len(colors)), colors, axis=0)
    return interp
