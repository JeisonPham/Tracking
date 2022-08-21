import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
import copy
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist


def fill_previous_times(path, missing_times):
    """
    First time must be known, times inbetween can be guessed from taking the previous value
    :param missing_times:
    :param path:
    :param times:
    :return:
    """

    for time in missing_times:
        temp = path[path[:, -1] == time - 1, :].copy()
        temp[:, -1] = time
        path = np.vstack([path, temp])

    return path[path[:, -1].argsort()]


def interpolate_missing_times(path, missing_times):
    """
    First and Last time must be known
    :param path:
    :param missing_times:
    :return:
    """
    interp = interp1d(path[:, -1], path[:, :-1], kind='linear', axis=0, bounds_error=True, assume_sorted=True)
    temp = np.hstack([interp(missing_times), np.array(missing_times).reshape(-1, 1)])
    path = np.vstack([path, temp])
    return path[path[:, -1].argsort()]


class Track:
    def __init__(self, label, starting, clock):
        """
        starting is an array of (x, y, u, v, theta) where x, y are position, u, v are velocity in each direction and vehicle angle
        """
        self.label = label
        self.path = [starting[:2]]
        self.time = [clock]
        self.age = 0

    def __repr__(self):
        return f"{self.label}: {self.path}"

    def add(self, next_point, time):
        self.path.append(next_point)
        self.time.append(time)

    def get_path(self, time_start, time_end, time_step):
        if self.time[0] > time_start or self.time[-1] < time_end:
            raise IndexError()

        points = []
        # interp = interp1d(self.time, self.path, kind='linear', axis=0, bounds_error=True, assume_sorted=True)
        times = np.arange(time_start, time_end + time_step, time_step)
        for time in times:
            if time in self.time:
                points.append(self.path[self.time.index(time)])
            else:
                points.append(points[-1])
        return np.array(points)

    def get_valid_paths(self, time_behind, time_ahead):
        times = np.arange(-time_behind, time_ahead + 2)
        # if len(times) > len(self.time):
        #     return None
        for time in self.time:
            offset = times + time
            inter = np.in1d(offset, self.time)
            if inter[0] and np.all(inter[4:]):
                inter2 = np.in1d(self.time, offset)
                path = np.array(self.path)[inter2].astype(float)
                path = np.hstack([path, np.array(self.time)[inter2].reshape(-1, 1)])
                path = interpolate_missing_times(path, offset[~inter])
                yield time, path[:, :-1], np.array(self.id)[inter2][-1], np.array(self.time)[inter2]


class KalmanFilterTrack(Track):
    def __init__(self, label, starting, clock):
        super().__init__(label, starting, clock)
        self.KF = KalmanFilter(dim_x=4, dim_z=2)
        self.KF.x = np.array(starting[:-1])
        self.KF.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]
                              ])
        self.KF.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]
                              ])
        self.KF.P *= 1000
        self.KF.R = 5

        self.path = []
        self.time = []
        self.id = []

        self.add(label, starting, clock)

    def add(self, id, next_point, time):
        super().add(next_point, time)
        self.id.append(id)
        self.KF.predict()
        self.KF.update(next_point[:2])

    @property
    def x(self):
        return self.KF.x

    @property
    def x_post(self):
        return self.KF.x_post


class KalmanFilterTracker:
    def __init__(self, distance_threshold=3, max_age=3):
        self.distance_threshold = distance_threshold
        self.id = 0
        self.max_age = max_age

        self.active_tracks = []
        self.inactive_tracks = []

    @property
    def tracks(self):
        return [*self.active_tracks, *self.inactive_tracks]

    def age_tracks(self):
        old_tracks = []
        for i in range(len(self.active_tracks)):
            self.active_tracks[i].age += 1
            if self.active_tracks[i].age == self.max_age:
                old_tracks.append(self.active_tracks[i])

        for i in old_tracks:
            self.inactive_tracks.append(i)
            self.active_tracks.remove(i)

    def update(self, detections, time):
        if len(detections) == 0:
            self.age_tracks()
        elif len(self.active_tracks) == 0:
            for det in detections:
                self.active_tracks.append(KalmanFilterTrack(self.id, det[1:], time))
                self.id += 1
        else:
            S = []
            adjusted_distance = detections[:, (1, 2)].astype(float) - detections[:, (3, 4)].astype(float)
            T1 = np.array([tracks.x[:2].astype(float) for tracks in self.active_tracks])
            F = cdist(adjusted_distance, T1[:, :2])

            N = len(detections)
            M = len(T1)
            for i in range(N):
                j = np.argmin(F[i])
                if j not in S and F[i, j] <= self.distance_threshold:
                    self.active_tracks[j].add(detections[i, 0], detections[i, 1:], time)
                    self.active_tracks[j].age = 0
                    S.append(j)
                    F[:, j] = np.inf
                else:
                    self.active_tracks.append(KalmanFilterTrack(self.id, detections[i, 1:], time))
                    self.id += 1

            for j in range(M):
                if j not in S:
                    if self.active_tracks[j].age < self.max_age:
                        self.active_tracks[j].age += 1

            old_tracks = []
            for i in range(len(self.active_tracks)):
                if self.active_tracks[i].age == self.max_age:
                    old_tracks.append(self.active_tracks[i])

            for i in old_tracks:
                self.inactive_tracks.append(i)
                self.active_tracks.remove(i)


class GroundTruthTracks:
    def __init__(self, id, df):
        self.id = id
        self.min_time = df['timestep_time'].min()
        self.max_time = df['timestep_time'].max()

        df = df.sort_values(by="timestep_time")

        t = df['timestep_time'].to_numpy()
        x = df[['vehicle_x', 'vehicle_y', 'vehicle_angle']].to_numpy()

        self.interp = interp1d(t, x, kind='linear', axis=0, bounds_error=True, assume_sorted=True)

    def usable(self, time):
        return self.min_time <= time <= self.max_time


class Tracker:
    def __init__(self, threshold=5, max_age=3):
        self.thresh = threshold
        self.age = max_age
        self.id = 0
        self.clock = -1

        self.labels = {}
        self.tracks = {}

    def get_tracks_at_time(self, time):
        points = []
        for label, track in self.tracks.items():
            pt = track.get_path_at_time(time)
            if not pt is None:
                points.append((pt, track.label))
        return points

    def __call__(self, T1, D, labels=None, clock=0):
        """
        Tracking Algorithm taken from https://arxiv.org/pdf/2006.11275.pdf

        :param prev_traj:

        T(t - 1) = {(p, v, c, q, id, a)} -> {(x, y, u, v, c, w, l, id, a)}
                     0, 1, 2, 3, 4, 5         0, 1, 2, 3, 4, 5, 6, 7, 8

        Tracked objects in the previous frame, with center p,
        ground plane velocity v, category label c,
        other bounding box attributes q, tracking id,
        and inactive age a (active tracks will have a = 0).

        :param detections:

        {(p, v, c, q)} -> {(x, y, u, v, c, w, l)}
          0, 1, 2, 3        0, 1, 2, 3, 4, 5, 6
        :return:

        T(t) = {(p, v, c, q, id, a)}
        """

        self.clock += 1

        T, S = list(), list()
        if len(D) == 0:
            if T1 is None or len(T1) == 0: return None

            t = T1[T1[:, 8] <= self.age]
            return t

        if T1 is None:
            t = np.hstack([D, np.zeros((len(D), 2))])
            for i in range(len(D)):
                if labels:
                    self.labels[self.id] = [(labels[i], clock)]
                    self.tracks[self.id] = Track(labels[i], D[i, :], clock)
                t[i, 7] = self.id
                self.id += 1
                t[i, 8] = 0

            return t

        adjusted_distance = D[:, (0, 1)] - D[:, (2, 3)]
        F = cdist(adjusted_distance, T1[:, (0, 1)])

        M, N = len(T1), len(D)
        for i in range(N):
            F_temp = F[i, :]
            mask = np.zeros(F_temp.size, dtype=bool)
            mask[S] = True
            a = np.ma.array(F_temp, mask=mask)
            j = np.argmin(a)
            t = T1[j, :].copy()
            t[:7] = D[i]
            t[-1] = 0
            if j not in S and F[i, j] <= self.thresh:
                if labels:
                    self.labels[t[7]].append((labels[i], clock))
                    self.tracks[t[7]].add(D[i, :], clock)
                T.append(t)
                S.append(j)
                # F[:, j] = np.inf

            else:
                if labels:
                    self.labels[self.id] = [(labels[i], self.clock)]
                    self.tracks[self.id] = Track(labels[i], D[i, :], clock)

                t[7] = self.id
                T.append(t)
                self.id += 1

        for j in range(M):
            if j not in S:
                if T1[j, -1] < self.age:
                    T1[j, -1] += 1
                    T1[j, (0, 1)] = T1[j, (0, 1)] + T1[j, (2, 3)]
                    T.append(T1[j, :].copy())

        return np.array(T)

    def AMOTA(self):

        MOTA = []
        group_by_time = {}
        for key, value in self.labels.items():
            for label, time in value:
                if time not in group_by_time:
                    group_by_time[time] = [None] * len(self.labels)

                group_by_time[time][key] = label

        prev = {}
        for key, value in group_by_time.items():
            FP, FN, IDS = 0, 0, 0
            for label in value:
                if label is None: continue
                if label not in prev:
                    prev[label] = value.index(label)
                elif prev[label] != value.index(label):
                    IDS += 1
                    prev[label] = value.index(label)

            MOTA.append(1 - (FP + FN + IDS) / (len(value) - value.count(None)))

        return np.average(MOTA)

    def Visualize(self, points):

        min_x, max_x = max(points[:, 0]), 0
        min_y, max_y = max(points[:, 2]), 0
        for label, track in self.tracks.items():
            path = np.array(track.path)

            min_x = min(min_x, min(path[:, 0]))
            min_y = min(min_y, min(path[:, 1]))

            max_x = max(max_x, max(path[:, 0]))
            max_y = max(max_y, max(path[:, 1]))
        # fig, ax = plt.subplots(figsize=(10, 10))
        plt.xlim([min_x, max_x])
        plt.ylim([min_y, max_y])
        for label, track in self.tracks.items():
            # ax.scatter(points[:, 0], points[:, 2], s=20, alpha=0.5)
            # ax.set_xlim([min_x, max_x])
            # ax.set_ylim([min_y, max_y])
            path = np.array(track.path)
            plt.plot(path[:, 0], path[:, 1])
        plt.show()

        # for label, track in self.tracks.items():
        #     future = np.array(track.predict(2))
        #     ax.plot(future[:, 0], future[:, 1], marker='o')
        #     break
        # plt.title("Paths")
        # plt.savefig("Paths.png")
        # plt.show()


if __name__ == "__main__":
    data = pd.read_csv("../Data/downtown_SD_10thru_50count_with_cad_id.csv")
    data['u'] = data.vehicle_speed * np.sin(data.vehicle_angle * np.pi / 180)
    data['v'] = data.vehicle_speed * np.cos(data.vehicle_angle * np.pi / 180)

    veh0 = data.loc[data.vehicle_id == 'veh0']
    other = data.loc[data.vehicle_id != 'veh0']

    tracker = KalmanFilterTracker(15, 3)

    time = 0
    ground_truth = {}
    for index, row in veh0.iterrows():
        x, y = row.loc[['vehicle_x', 'vehicle_y']].to_numpy()
        u, v = row.loc[['u', 'v']].to_numpy()

        other_veh = other.loc[(other["vehicle_x"] > x - 60) & (other["vehicle_x"] < x + 60)
                              & (other["vehicle_y"] > y - 60)
                              & (other["vehicle_y"] < y + 60) & (other["timestep_time"] == row['timestep_time'])]

        D = other_veh[['vehicle_id', 'vehicle_x', 'vehicle_y', 'u', 'v', "vehicle_angle"]].to_numpy()
        for d in D:
            if d[0] not in ground_truth:
                ground_truth[d[0]] = []
            ground_truth[d[0]].append(d[1:].astype(float))
        tracker.update(D, time)
        time += 1

    # for track in tracker.tracks:
    #     path = np.array(track.path).astype(float)
    #     plt.plot(path[:, 0], path[:, 1])
    #     plt.show()

    for veh, track in ground_truth.items():
        track = np.array(track)
        plt.plot(track[:, 0], track[:, 1], alpha=0.5)
    plt.show()

    for track in tracker.tracks:
        path = np.array(track.path).astype(float)
        plt.plot(path[:, 0], path[:, 1])
    plt.show()
