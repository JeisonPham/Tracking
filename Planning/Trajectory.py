import numpy as np
from scipy.optimize import curve_fit


def x_clothoid(L, R, l):
    return l - np.power(l, 5) / (40 * np.power(R * L, 2)) + np.power(l, 9) / (3456 * np.power(R * L, 4))


def y_clothoid(L, R, l):
    return np.power(l, 3) / (6 * R * L) - np.power(l, 7) / (336 * np.power(R * L, 3))


class Trajectory(object):
    def __init__(self, start, tol=10):


        self.cluster = np.asarray([start])
        self.tol = tol
        self.i = len(start)

    @property
    def mean(self):
        # cluster = []
        # for c in self.cluster:
        #     cluster.append(c)
        temp = np.mean(self.cluster, axis=0)
        temp[(temp[:, 0] > 255)] = 255
        temp[(temp[:, 0] < 0)] = 0
        temp[(temp[:, 1] > 255)] = 255
        temp[(temp[:, 1] < 0)] = 0
        return temp.astype(int)

    def distance(self, other_traj):
        distance = np.linalg.norm(self.cluster - other_traj, axis=2)
        if np.max(distance) > self.tol:
            return np.nan
        return np.max(distance)

    def angle(self, other_traj):
        min_angle = 360
        for traj in self.cluster:
            i = min(len(traj), len(other_traj))
            angle = 0
            for j in range(i):
                traj_angle = np.arctan2(traj[j, 1] - traj[0, 1], traj[j, 0] - traj[0, 0])
                other_angle = np.arctan2(other_traj[j, 1] - other_traj[0, 1], other_traj[j, 0] - other_traj[0, 0])

                angle = max(angle, abs(traj_angle - other_angle))
            min_angle = min(min_angle, angle)
            if angle > np.pi / 8:
                return angle
        return min_angle

    def update(self, other_traj):
        result = not np.isnan(self.distance(other_traj))
        if result:
            self.cluster = np.append(self.cluster, other_traj.reshape(1, 16, 2), axis=0)
        return result
