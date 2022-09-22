import scipy.spatial.distance
import torch
import torch.utils.data as torchdata
import os
import pandas as pd
import open3d as o3d
from scipy.interpolate import interp1d
from util import *
import pickle
from glob import glob
import re


class RadarDataset(torchdata.Dataset):

    def __init__(self, fileLocation, carFile, cloudFile, ego_only, flip_aug, is_train,
                 t_spacing, only_y, no_interp=False, pre_load=False):
        self.is_train = is_train
        self.ego_only = ego_only
        self.t_spacing = t_spacing  # seconds
        self.stretch = 70.0  # meters
        self.only_y = only_y
        self.flip_aug = flip_aug
        self.local_ts = np.arange(0.25, 4.1, 0.25)  # seconds

        self.no_interp = no_interp
        self.pre_load = pre_load

        df = pd.read_csv(os.path.join(fileLocation, carFile))

        if df["vehicle_angle"].max() > 2 * np.pi or df["vehicle_angle"].min() < 0:
            df["vehicle_angle"] = df["vehicle_angle"] * np.pi / 180

        self.df = df
        self.vehicle_names = list(df["vehicle_id"].unique())

        self.dx, self.bx, (self.nx, self.ny) = get_grid([-17.0, -38.5,
                                                         60.0, 38.5],
                                                        [0.3, 0.3])

        # if not self.pre_load:
        self.data = self.compile_data()
        self.ixes = self.get_ixes()

        # with open("/media/jason/easystore/Colllab-Radar-Data/pre_generated_data.pkl", "wb") as file:
        #     pickle.dump({"ixes": self.ixes, "data": self.data}, file)
        # # else:
        #     with open("/media/jason/easystore/Colllab-Radar-Data/pre_generated_data.pkl", "rb") as file:
        #         p = pickle.load(file)
        #     self.data = p["data"]
        #     self.ixes = p['ixes']

        pc = o3d.io.read_point_cloud(os.path.join(fileLocation, cloudFile))
        points = np.asarray(pc.points)

        points = points[points[:, 0] < 390]

        min_x = min(points[:, 0])
        min_y = max(points[:, 2])
        offset = np.array([-min_x, 0, -min_y])
        points = points + offset
        points[:, 2] = -points[:, 2]
        self.points = points

    def compile_data(self):
        scene2data = {}
        print("Compiling Data")
        for name in tqdm(self.vehicle_names[:40]):
            veh = self.df[self.df["vehicle_id"] == name]

            for index, row in veh.iterrows():
                scene = row["vehicle_id"]
                if scene not in scene2data:
                    scene2data[scene] = {}
                    scene2data[scene]["ego"] = {'traj': [], 'w': 1.73, 'l': 4.084, 'k': 'ego'}

                angle = row["vehicle_angle"]
                # rot = get_rot(angle)
                # rot = np.arctan2(rot[1, 0], rot[0, 0])
                scene2data[scene]['ego']['traj'].append({
                    'x': row['vehicle_x'],
                    'y': row['vehicle_y'],
                    'hcos': np.cos(angle),
                    'hsin': np.sin(angle),
                    't': row['timestep_time']
                })

                other_instances = self.df.loc[(self.df["timestep_time"] == row["timestep_time"])
                                              & (self.df['vehicle_x'] <= row['vehicle_x'] + 60) & (
                                                          self.df['vehicle_x'] >= row['vehicle_x'] - 60)
                                              & (self.df['vehicle_y'] <= row['vehicle_y'] + 60) & (
                                                          self.df['vehicle_y'] >= row['vehicle_y'] - 60)]
                #     print(other_instances["vehicle_angle"])
                for ii, rr in other_instances.iterrows():
                    if rr["vehicle_id"] == scene:
                        continue

                    instance = rr["vehicle_id"]
                    if instance not in scene2data[scene]:
                        scene2data[scene][instance] = {'traj': [], 'w': 1.73, 'l': 4.084, 'k': rr["vehicle_type"]}

                    angle = rr["vehicle_angle"]
                    # rot = get_rot(angle)
                    # rot = np.arctan2(rot[1, 0], rot[0, 0])
                    scene2data[scene][instance]['traj'].append({
                        'x': rr["vehicle_x"],
                        'y': rr["vehicle_y"],
                        'hcos': np.cos(angle),
                        'hsin': np.sin(angle),
                        't': rr['timestep_time']
                    })

        return self.post_process(scene2data)

    def post_process(self, data):
        scene2info = {}
        print("Post Process")
        for scene in tqdm(data):
            scene2info[scene] = {}
            for name in data[scene]:
                info = {}
                t = [row['t'] for row in data[scene][name]['traj']]
                x = [[row['x'], row['y'], row['hcos'], row['hsin']] for row in data[scene][name]['traj']]

                if t[-1] == t[0]:
                    t.append(t[0] + 0.02)
                    x.append([val for val in x[-1]])

                info['interp'] = interp1d(t, x, kind='linear', axis=0,
                                          copy=False, bounds_error=True,
                                          assume_sorted=True)
                info['lw'] = [data[scene][name]['l'], data[scene][name]['w']]
                info['k'] = data[scene][name]['k']
                info['tmin'] = t[0]
                info['tmax'] = t[-1]

                scene2info[scene][name] = info
        return scene2info

    def get_ixes(self):
        ixes = []
        min_distance = 0.2
        print("Generate ixes")
        for scene in tqdm(self.data):
            if not self.ego_only:
                names = [name for name in self.data[scene]]
            else:
                names = ["ego"]

            for name in names:
                # if not self.no_interp:
                ts = np.arange(self.data[scene][name]['tmin'],
                               self.data[scene][name]['tmax'] - self.local_ts[-1],
                               self.t_spacing)
                # else:
                #     ts = np.arange(self.data[scene][name]['tmin'],
                #                    self.data[scene][name]['tmax'] - self.local_ts[-1],
                #                    1)

                for t in ts:
                    interp = self.data[scene][name]['interp']
                    dist = np.linalg.norm(interp(t + self.local_ts[-1])[:2] - interp(t)[:2])
                    if name == "ego" or dist > min_distance:
                        ixes.append((scene, name, t))
        return ixes

    def get_ixes2(self):
        ixes = []
        files = glob('/media/jason/easystore/Colllab-Radar-Data/Ground_Truth/*')
        for file in files:
            file = file.split("/")[-1]
            group = re.search(r'(\w+)_(.+)_(.+)\.npy$', file).group(1, 2, 3)
            ixes.append((group[0], group[1], group[2]))
        return ixes

    def render(self, scene, name, t0):
        center = self.data[scene][name]['interp'](t0)

        centerlw = self.data[scene][name]['lw']

        objs = np.array([row['interp'](t0)
                         for na, row in self.data[scene].items()
                         if na != name and row['tmin'] <= t0 <= row['tmax']])

        dist = False
        for obj in objs:
            if np.linalg.norm(center[:2] - obj[:2]) <= 20:
                dist = True

        lws = np.array([row['lw'] for na, row in self.data[scene].items()
                        if na != name and row['tmin'] <= t0 <= row['tmax']])

        if len(objs) == 0:
            lobjs = np.zeros((0, 4))
        else:
            lobjs = objects2frame(objs[np.newaxis, :, :], center)[0]

        tgt = self.data[scene][name]['interp'](t0 + self.local_ts)
        ltgt = objects2frame(tgt[np.newaxis, :, :], center)[0]

        # create image of other objects
        obj_img = np.zeros((self.nx, self.ny))
        for box, lw in zip(lobjs, lws):
            pts = get_corners(box, lw)
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(obj_img, [pts], 1.0)

        # create image of ego
        center_img = np.zeros((self.nx, self.ny))
        pts = get_corners([0.0, 0.0, 1.0, 0.0], centerlw)
        pts = np.round(
            (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
        ).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]
        cv2.fillPoly(center_img, [pts], 1.0)

        # create image of map
        angle = -np.arctan2(center[3], center[2]) + np.pi / 2
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        area = get_area(self.points, center[0], center[1], 70)
        pts = np.dot(area - np.array([center[0], center[1]]), rot)
        pts = np.round(
            (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
        ).astype(np.int32)
        pts[:, [0, 1]] = pts[:, [1, 0]]
        map_image = np.zeros((self.nx, self.ny))
        for pt in pts:
            cv2.circle(map_image, (pt[0], pt[1]), 1, (255, 0, 0), thickness=1)

        map_image /= 255.0

        # return np.stack([center_img, obj_img, map_image]), self.get_tgt(ltgt), center
        lane, road_div = np.zeros((self.nx, self.ny)), np.zeros((self.nx, self.ny))
        x = np.stack([map_image, lane, road_div, obj_img, center_img])
        return x , self.get_tgt(ltgt), center

    def get_tgt(self, ltgt):
        pts = np.round(
            (ltgt[:, :2] - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
        ).astype(np.int32)
        tgt = np.zeros((ltgt.shape[0], self.nx, self.ny))

        pts = np.concatenate((
            np.arange(ltgt.shape[0], dtype=np.int32)[:, np.newaxis],
            pts), 1)
        kept = np.logical_and(0 <= pts[:, 1], pts[:, 1] < self.nx)
        kept = np.logical_and(kept, 0 <= pts[:, 2])
        kept = np.logical_and(kept, pts[:, 2] < self.ny)

        pts = pts[kept]
        tgt[pts[:, 0], pts[:, 1], pts[:, 2]] = 1.0

        return tgt

    def __len__(self):
        return len(self.ixes)

    def __getitem__(self, index):
        scene, name, t0 = self.ixes[index]

        # if not self.pre_load:

        if self.only_y:
            center = self.data[scene][name]['interp'](t0)
            tgt = self.data[scene][name]['interp'](t0 + self.local_ts)
            ltgt = objects2frame(tgt[np.newaxis, :, :], center)[0]
            y = self.get_tgt(ltgt)
            # if np.random.rand() > 0.5:
            #     y = np.flip(y, 2).copy()
            return torch.Tensor(y)

        x, y, center = self.render(scene, name, float(t0))

        # if dist:
        #     print("saving")
        #     np.save(f"/media/jason/easystore/Colllab-Radar-Data/Ground_Truth/{scene}_{name}_{t0}.npy", x)
        #     np.save(f"/media/jason/easystore/Colllab-Radar-Data/Predictions/{scene}_{name}_{t0}.npy", y)

        # else:
        # x = np.load(f"/media/jason/easystore/Colllab-Radar-Data/Ground_Truth/{scene}_{name}_{t0}.npy")
        # y = np.load(f"/media/jason/easystore/Colllab-Radar-Data/Predictions/{scene}_{name}_{t0}.npy")

        return torch.Tensor(x), torch.Tensor(y)


def compile_data(file_location, vehicle_file, map_file,
                 ego_only, flip_aug, is_train, t_spacing, only_y, no_interp=False, pre_load=False,
                 train_size=0.75, batch_size=16, num_workers=10, ):
    dataset = RadarDataset(file_location, vehicle_file, map_file, ego_only, flip_aug,
                           is_train, t_spacing, only_y, no_interp=no_interp, pre_load=pre_load)

    print(f"Dataset is: {len(dataset)}")

    n = len(dataset)
    split = int(n * train_size)
    idx = list(range(n))

    traindata = torch.utils.data.Subset(dataset, idx[:split])
    valdata = torch.utils.data.Subset(dataset, idx[split:])

    trainloader = torchdata.DataLoader(traindata, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers)

    valloader = torchdata.DataLoader(valdata, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers)

    return trainloader, valloader


def pre_generate_dataset(file_location, vehicle_file, map_file,
                         ego_only, flip_aug, is_train, t_spacing, only_y, no_interp=False,
                         train_size=0.75, batch_size=16, num_workers=10):
    dataset = RadarDataset(file_location, vehicle_file, map_file, ego_only, flip_aug,
                           is_train, t_spacing, only_y, no_interp)

    print(len(dataset))

    loader = iter(dataset)

    for x, y in loader:
        print(x.shape)
