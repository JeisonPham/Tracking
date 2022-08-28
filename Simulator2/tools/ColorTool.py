import numpy as np
import scipy.interpolate
from typing import List


class Color:
    WHITE = [1.0, 1.0, 1.0]
    SILVER = [0.75, 0.75, 0.75]
    GRAY = [0.5, 0.5, 0.5]
    BLACK = [0., 0., 0.]
    RED = [1.0, 0.0, 0.0]
    MAROON = [0.5, 0., 0.]
    YELLOW = [1.0, 1.0, 0.]
    OLIVE = [0.5, 0.5, 0.]
    LIME = [0., 1.0, 0.]
    GREEN = [0., 0.5, 0.]
    AQUA = [0., 1.0, 1.]
    TEAL = [0., 0.5, 0.5]
    BLUE = [0., 0., 1.]
    NAVY = [0., 0., 0.5]
    FUCHSIA = [1., 0., 1.]
    PURPLE = [0.5, 0., 0.5]

    @staticmethod
    def gradient_helper(colors: List[List[float]], spacing=None):
        colors = np.asarray(colors)
        if spacing is None:
            x = np.array(list(range(colors.shape[0]))) / colors.shape[0]
        else:
            x = spacing

        interp = scipy.interpolate.interp1d(x, colors, axis=0, fill_value="extrapolate")

        return interp


if __name__ == "__main__":
    gradient = Color.gradient_helper(colors=[Color.WHITE, Color.BLUE, Color.GREEN])
    print(gradient(0.4))
