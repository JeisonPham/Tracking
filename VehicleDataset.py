import pandas as pd
import numpy as np


class VehicleDataset:
    def __init__(self, file):
        self.data = pd.read_csv(file)
        self.data['u'] = self.data.vehicle_speed * np.sin(self.data.vehicle_angle * np.pi / 180)
        self.data['v'] = self.data.vehicle_speed * np.cos(self.data.vehicle_angle * np.pi / 180)

    def get_nearby_vehicles(self, ego_name, time):
        x, y, u, v = self.data.loc[(self.data['vehicle_id'] == ego_name) & (self.data['timestep_time'] == time),
                                   ['vehicle_x', 'vehicle_y', "u", "v"]].to_numpy()[0]

        other = self.data.loc[self.data.vehicle_id != ego_name]
        other_veh = other.loc[(other["vehicle_x"] > x - 60) & (other["vehicle_x"] < x + 60)
                              & (other["vehicle_y"] > y - 60)
                              & (other["vehicle_y"] < y + 60) & (
                                      other["timestep_time"] == time)]

        D = other_veh[['vehicle_id', 'vehicle_x', 'vehicle_y', 'u', 'v', "vehicle_angle"]].to_numpy()
        if len(D) > 0:
            D[:, 1:3] += np.random.normal(0, 0, (len(D), 2))
        return D

    def get_vehicle_GT(self, ego_name, times):
        data = self.data[(self.data.vehicle_id == ego_name) & (self.data.timestep_time.isin(times))]
        return data
