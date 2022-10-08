from Planning import PolygonDataset
import numpy as np
import pandas as pd
import os
import sys
from Planning.util import *
from collab_radar_eval.utils.collab_dataset_utils.dataset_utils import collab_dataset

sys.path.append("Planning")

gt_folder_name = "downtown_SD_10thru_50count_labels"
bev_folder_name = "downtown_SD_10thru_50count_80m_doppler_tuned"


class RadarDataset(PolygonDataset):
    def __init__(self, car_file, trajectory_file=None, radar_dataset="", ego_only=True, only_y=False, t_spacing=0.25,
                 split_file=None, raw_output=False, image_output=False, only_speed=False):
        self.ego_only = ego_only
        self.t_spacing = t_spacing
        self.only_y = only_y
        self.local_ts = np.arange(0.25, 4.1, 0.25)
        self.only_speed = only_speed
        self.raw_output = raw_output

        self.collab_dataset = collab_dataset(radar_dataset, gt_folder_name, bev_folder_name)
        df = pd.read_csv(car_file)
        if df["vehicle_angle"].max() > 2 * np.pi or df["vehicle_angle"].min() < 0:
            df["vehicle_angle"] = df["vehicle_angle"] * np.pi / 180

        self.df = df

        with open(os.path.join(radar_dataset, "training_txt_files", bev_folder_name, "valid_files_train.txt"), 'r') as file:
            training = file.readlines()

        with open(os.path.join(radar_dataset, "training_txt_files", bev_folder_name, "valid_files_test.txt"), 'r') as file:
            testing = file.readlines()

        self.radar_data = training + testing
        self.vehicle_names = set([x.split("_")[2] for x in self.radar_data])



        # self.dx, self.bx, (self.nx, self.ny) = get_grid([-17.0, -38.5,
        #                                                  60.0, 38.5],
        #                                                 [0.3, 0.3])

        self.data = self.compile_data()
        self.ixes = self.get_ixes()

        valid_ixes = []
        for x in self.radar_data:
            plot, data, name, time = x.split("_")
            time = float(time)
            if (name, "ego", time) in self.ixes:
                valid_ixes.append((name, "ego", time))

        self.ixes = valid_ixes

    def render(self, scene, name, t0):
        veh = int(scene[3:])
        img = self.collab_dataset.get_image(timestamp=t0, veh_id=veh)
        extrinsic_ego2world = self.collab_dataset.get_extrinsic(timestamp=t0, veh_id=veh)

        tgt = self.data[scene][name]['interp'](t0 + self.local_ts)
        for i, pos in enumerate(tgt):
            tgt[i] = np.dot(np.linalg.norm(extrinsic_ego2world), tgt)

        print(img, tgt)

