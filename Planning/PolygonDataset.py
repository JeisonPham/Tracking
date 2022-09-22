import torch.utils.data as torchdata
import pandas as pd
from tqdm import tqdm
from util import *
import json
import numpy as np
from scipy.interpolate import interp1d
from glob import glob
import re
import json
import random
import pickle


class PolygonDataset(torchdata.Dataset):
    def __init__(self, car_file, polygon_file, trajectory_file=None, ego_only=False, only_y=False, t_spacing=0.25,
                 split_file=None, image_output=False):
        self.carFile = car_file
        self.ego_only = ego_only
        self.t_spacing = t_spacing
        self.only_y = only_y
        self.local_ts = np.arange(0.25, 4.1, 0.25)

        df = pd.read_csv(car_file)
        if df["vehicle_angle"].max() > 2 * np.pi or df["vehicle_angle"].min() < 0:
            df["vehicle_angle"] = df["vehicle_angle"] * np.pi / 180

        self.df = df
        self.vehicle_names = list(df["vehicle_id"].unique())

        self.dx, self.bx, (self.nx, self.ny) = get_grid([-17.0, -38.5,
                                                         60.0, 38.5],
                                                        [0.3, 0.3])

        self.data = self.compile_data()
        self.ixes = self.get_ixes()

        with open(polygon_file, "r") as file:
            self.polygon = json.load(file)

        if split_file is not None:

            with open(split_file, 'r') as file:
                distribution = json.load(file)

            self.ixes = []

            for v in distribution.values():
                self.ixes += v

            random.seed(42)
            random.shuffle(self.ixes)

        if image_output:
            self.tgt = self.get_tgt_image
        else:
            self.tgt = self.get_tgt

        self.speed_dist = [0] * 25

        if trajectory_file is not None:
            with open(trajectory_file, 'rb') as file:
                self.trajectory_set = pickle.load(file)

    def __len__(self):
        return len(self.ixes)

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

        polygon_list = {}
        for poly_type, value in self.polygon.items():
            if isinstance(value, list): continue
            polygon_list[poly_type] = {}
            for key, shape in value.items():
                if poly_type == 'lane_markings' and key in polygon_list['lane']:
                    for i in range(len(shape)):
                        polygon_list[poly_type][key + str(i)] = np.asarray(shape[i])
                elif poly_type == 'lane_markings':
                    continue
                else:
                    shape = np.asarray(shape)

                    if any(np.linalg.norm(shape - center[:2], axis=1) <= 60):
                        polygon_list[poly_type][key] = np.asarray(shape)

        map_img = np.zeros((self.nx, self.ny))

        def draw_polygon(shapes, image, fill=False):
            angle = -np.arctan2(center[3], center[2]) + np.pi / 2
            rot = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
            for key, poly in shapes.items():
                poly = np.dot(poly - np.array([center[0], center[1]]), rot)
                pts = np.round(
                    (poly - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
                ).astype(np.int32)
                pts = np.clip(pts, 0, self.nx)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                if fill:
                    cv2.fillPoly(image, [pts], color=1.0)
                else:
                    cv2.polylines(image, [pts], isClosed=False, color=1.0)

        draw_polygon(polygon_list['junction'], map_img, fill=True)
        draw_polygon(polygon_list['lane'], map_img, fill=True)

        road_div = np.zeros((self.nx, self.ny))
        draw_polygon(polygon_list['junction'], road_div)
        draw_polygon(polygon_list['lane'], road_div)

        lane_div = np.zeros((self.nx, self.ny))
        x = np.stack([map_img, lane_div, road_div, obj_img, center_img])
        tgt, direction = self.tgt(ltgt)
        return x, tgt, direction

    def get_tgt(self, ltgt):
        pts = np.round(
            (ltgt[:, :2] - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
        ).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]
        temp = pts[-1] - pts[0]

        speed = np.linalg.norm(pts[0] - pts[15]) / len(pts)
        self.speed_dist[int(speed)] += 1

        temp = temp.flatten()
        angle = np.rad2deg(np.arctan2(temp[1], temp[0]))
        lower_bounds = 75
        upper_bounds = 105

        if lower_bounds < angle < upper_bounds:
            dire = 0
        elif -lower_bounds <= angle <= lower_bounds:
            dire = 1
        elif -upper_bounds < angle < -lower_bounds:
            dire = 2
        else:
            dire = 3

        return pts, (speed, dire)

    def get_tgt_image(self, ltgt):
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

        return tgt, 0

    def get_trajectory_index(self, other_traj, speed, direction):


        trajectory_set = self.trajectory_set[speed][direction]

        keys = list(trajectory_set.keys())
        distances = [trajectory_set[key].distance(other_traj) for key in keys]
        ii = np.nanargmin(distances)

        return keys[ii], trajectory_set[keys[ii]]

    def __getitem__(self, index):
        scene, name, t0 = self.ixes[index]

        if self.only_y:
            center = self.data[scene][name]['interp'](t0)
            tgt = self.data[scene][name]['interp'](t0 + self.local_ts)
            ltgt = objects2frame(tgt[np.newaxis, :, :], center)[0]
            y, direction = self.tgt(ltgt)
            # if np.random.rand() > 0.5:
            #     y = np.flip(y, 2).copy()
            return (scene, name, t0), torch.Tensor(y), direction

        x, y, (speed, direction) = self.render(scene, name, float(t0))

        if speed <= 5:
            speed = "less_than_5"
        else:
            speed = "greater_than_5"

        index, trajectory = self.get_trajectory_index(y, speed, direction)

        trajectory_set = self.trajectory_set[speed][direction]

        idxes = list(self.trajectory_set[speed][direction])
        idxes.remove(index)

        negative_idxes = random.sample(idxes, 18)
        index_column = np.arange(trajectory.mean.shape[0], dtype=int).reshape(-1, 1)
        negative_trajectory = [np.hstack([index_column.copy(), trajectory_set[index].mean]) for index in
                               negative_idxes]
        negative_trajectory.append(np.hstack([index_column.copy(), trajectory.mean]))
        negative_trajectory = np.asarray(negative_trajectory)

        distance = np.linalg.norm(negative_trajectory[:, :, 1:] - trajectory.mean, axis=2)

        return torch.Tensor(x), torch.Tensor(y), (distance, negative_trajectory, speed, direction)


if __name__ == "__main__":
    from explore import create_mask, viz_masks, create_trajectory_set

    data = PolygonDataset(r"F:\2022-09-12-16-57-35\test.csv",
                          r"F:\Radar Reseach Project\Tracking\SumoNetVis\polygons.json", only_y=True,
                          image_output=False)

    # create_mask(data, "mask.json")
    # viz_masks("mask.json")
    # print(len(data))
    it = iter(data)
    directions = {i: [] for i in range(4)}
    for x, y, dire in it:
        directions[dire[1]].append(x)

    directions[1] = random.sample(directions[1], len(directions[3]))

    with open("distribution.json", 'w') as file:
        json.dump(directions, file)
