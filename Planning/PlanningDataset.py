from RadarDataset import RadarDataset
from Trajectory import Trajectory
import pickle
import open3d as o3d
import torch
import numpy as np
from util import *
import random


class PlanningDataset(RadarDataset):
    def __init__(self, trajectory_file="", *args, **kwargs):
        super(PlanningDataset, self).__init__(*args, **kwargs)

        with open(trajectory_file, "rb") as file:
            self.trajectory_set = pickle.load(file)

        self.keys = list(self.trajectory_set.keys())
        self.values = list(self.trajectory_set.values())

    def get_trajectory_index(self, other_traj):
        keys = list(self.trajectory_set.keys())
        distances = [self.trajectory_set[key].distance(other_traj) for key in keys]
        # if np.all(np.isnan(distances)):
        #     distances = [x.angle(other_traj) for x in self.values]
        ii = np.nanargmin(distances)

        return keys[ii], self.trajectory_set[keys[ii]]

    def __getitem__(self, item):
        scene, name, t0 = self.ixes[item]

        x, y, center = self.render(scene, name, float(t0))

        idxes = np.array(np.where(y == 1))[1:, :].T

        index, trajectory = self.get_trajectory_index(idxes)

        idxes = list(self.trajectory_set.keys())
        idxes.remove(index)

        negative_idxes = random.sample(idxes, 18)
        index_column = np.arange(trajectory.mean.shape[0], dtype=int).reshape(-1, 1)
        negative_trajectory = [np.hstack([index_column.copy(), self.trajectory_set[index].mean]) for index in negative_idxes]
        negative_trajectory.append(np.hstack([index_column.copy(), trajectory.mean]))
        negative_trajectory = np.asarray(negative_trajectory)

        distance = np.linalg.norm(negative_trajectory[:, :, 1:] - trajectory.mean, axis=2)

        return torch.Tensor(x), torch.Tensor(y), (distance, negative_trajectory)

if __name__ == "__main__":
    PD = PlanningDataset(trajectory_file="../Planning/Trajectory_set.pkl",
                         fileLocation = "../Data",
                         carFile = "downtown_SD_10thru_50count_with_cad_id.csv",
                         cloudFile = "downtown_SD_10_7.ply", ego_only=False, flip_aug=False, is_train=False, t_spacing=0.25, only_y =False)

    for i, (x, y, GT) in enumerate(iter(PD)):
        render_observations_and_traj(x, y, GT[1].astype(int))
        plt.savefig(f"../Planning/Visuals/{i:05d}.png")
        plt.clf()
