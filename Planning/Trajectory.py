import numpy as np


class Trajectory(object):
    def __init__(self, start, tol=5):
        self.cluster = [start]
        self.tol = tol
        self.i = len(start)

    @property
    def mean(self):
        cluster = []
        for c in self.cluster:
            cluster.append(c[:self.i, :])
        temp = np.zeros((16, 2))
        temp[:self.i, :] = np.mean(cluster, axis=0)
        return temp.astype(int)

    def distance(self, other_traj):
        for traj in self.cluster:
            i = min(len(traj), len(other_traj))
            dist = np.linalg.norm(traj[:i, :] - other_traj[:i, :], axis=1)
            if max(dist) > self.tol:
                return np.nan
        return max(dist)

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
            self.i = min(self.i, len(other_traj))
            self.cluster.append(other_traj)
        return result
