import numpy as np
from scipy.interpolate import interp1d


def object_to_ego(ego, obj):
    ego_angle = np.deg2rad(ego[-1])
    c, s = np.cos(ego_angle), np.sin(ego_angle)
    rot = np.array([[c, -s], [s, c]])

    positions = np.dot(obj[:, :2] - ego[:2], rot)
    velocity = -positions[:-1, :] + positions[1:, :]
    positions = positions[:-1, :]
    angle = obj[1:, -1] - ego[2]
    angle = angle.reshape(-1, len(angle)).T

    return np.hstack((positions, velocity, angle))


def remove_detection(path, index):
    path[index] = np.nan
    return path


def replace_with_previous(path, index):
    path[index] = path[index - 1]
    return path


def interpolate_missing(path, index):
    previous = path[:5, :]
    previous = np.delete(previous, index, axis=0)
    indexes = np.arange(0, 5, 1)
    indexes = np.delete(indexes, index)

    interp = interp1d(indexes, previous, kind='linear', axis=0, fill_value="extrapolate", assume_sorted=True)

    path[index] = interp(index)

    return path
