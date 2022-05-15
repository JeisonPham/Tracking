import numpy as np
import math


def ADE_K(output, actual, k):
    pred_len = actual.shape[0]
    ade = float(
        sum(
            math.sqrt(
                (actual[i, 0] - output[i, 0]) ** 2
                + (actual[i, 1] - output[i, 1]) ** 2
            )
            for i in range(pred_len)
        )
        / pred_len
    )
    return ade


def ADE_I(output, actual):
    return np.linalg.norm(output - actual, axis=1)


def FDE(output, actual):
    pred_len = actual.shape[0]
    fde = math.sqrt(
        (actual[-1, 0] - output[-1, 0]) ** 2
        + (actual[-1, 1] - output[-1, 1]) ** 2
    )
    return fde
