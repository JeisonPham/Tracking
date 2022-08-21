import torch.utils.data as torchdata
import pandas as pd
from TrackingModule.Tracking import GroundTruthTracks
import numpy as np
from tqdm import tqdm


class TrackingLoader(torchdata.Dataset):
    def __init__(self, csv_file, previous_steps, future_steps, max_tracks=20, step=1, visualize_mode=False):
        self.previous_steps = previous_steps
        self.future_steps = future_steps
        self.tracks = []
        self.visualize_mode = visualize_mode
        self.step = step

        data = pd.read_csv(csv_file)
        data['u'] = data.vehicle_speed * np.sin(data.vehicle_angle * np.pi / 180) + np.random.normal(2, 2,
                                                                                                     len(data.index))
        data['v'] = data.vehicle_speed * np.cos(data.vehicle_angle * np.pi / 180) + np.random.normal(2, 2,
                                                                                                     len(data.index))

        data = data.dropna()
        for veh in set(data.vehicle_id):
            if max_tracks < int(veh[3:]):
                continue
            df = data[data.vehicle_id == veh]
            track = GroundTruthTracks(veh, df)
            self.tracks.append(track)

        self.idxes = self.get_indexes(self.tracks)

    def get_indexes(self, tracks):
        idex = []
        for track in tqdm(tracks):
            for time in range(int(track.min_time), int(track.max_time)):
                ego_info = track.interp(time)
                ego_angle = np.deg2rad(ego_info[2])
                c, s = np.cos(ego_angle), np.sin(ego_angle)
                rot = np.array([[c, -s], [s, c]])

                for other_track in tracks:
                    if track == other_track:
                        continue

                    # looks at one position behind so velocity can be calculated
                    # times = np.array(range(-self.previous_steps, self.future_steps + self.step, self.step)) + time
                    times = np.arange(-self.previous_steps, self.future_steps + self.step * 2, self.step)
                    times += time
                    if other_track.usable(times[0]) and other_track.usable(times[-1]):
                        other_info = other_track.interp(times)
                        positions = np.dot(other_info[:, :2] - ego_info[:2], rot)

                        if np.all(np.linalg.norm(positions[:, (0, 1)]) > 150):
                            continue
                        velocity = -positions[:-1, :] + positions[1:, :]
                        positions = positions[:-1, :]
                        angle = other_info[1:, 2] - ego_info[2]
                        angle = angle.reshape(-1, len(angle)).T
                        idex.append((track.id, other_track.id, time,
                                     np.hstack((positions, velocity, angle, times[:-1].reshape(-1, 1)))))

        if self.visualize_mode:
            self.data = {}
            for point in idex:
                key = (point[0], point[1], point[2])
                if key not in self.data:
                    self.data[key] = []
                self.data[key].append(point[-1])
            return list(self.data.keys())
        else:
            return idex

    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, index):
        if self.visualize_mode:
            key = self.idxes[index]
            return key, self.data[key]
        else:
            veh, other_veh, time, info = self.idxes[index]
            split = int(self.previous_steps / self.step) + 1
            return info[:split, :-1], info[split:, :2]


if __name__ == "__main__":
    data = TrackingLoader("../Data/downtown_SD_10thru_50count_with_cad_id.csv", 5, 3)
    print(len(data))
