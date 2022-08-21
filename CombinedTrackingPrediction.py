import pandas as pd
from TrackingModule.Tracking import KalmanFilterTracker
import matplotlib.pyplot as plt
from TrackingModule.utils import *
import torch
from PredictionModule.model import MLP
from PredictionModule.metrics import *
from VehicleDataset import VehicleDataset

if __name__ == "__main__":
    data = pd.read_csv("Data/downtown_SD_10thru_50count_with_cad_id.csv")
    data['u'] = data.vehicle_speed * np.sin(data.vehicle_angle * np.pi / 180)
    data['v'] = data.vehicle_speed * np.cos(data.vehicle_angle * np.pi / 180)

    dataset = VehicleDataset(file="Data/downtown_SD_10thru_50count_with_cad_id.csv")

    random_threshold = 0.75
    np.random.seed(42)


    ades = []
    for veh in set(data['vehicle_id']):
        if isinstance(veh, str) and int(veh[3:]) <= 100:

            veh0 = data.loc[data.vehicle_id == veh]
            other = data.loc[data.vehicle_id != veh]

            tracker = KalmanFilterTracker(15, 3)

            time = 0
            ground_truth = {}
            for index, row in veh0.iterrows():
                # x, y = row.loc[['vehicle_x', 'vehicle_y']].to_numpy()
                # u, v = row.loc[['u', 'v']].to_numpy()
                #
                # other_veh = other.loc[(other["vehicle_x"] > x - 60) & (other["vehicle_x"] < x + 60)
                #                       & (other["vehicle_y"] > y - 60)
                #                       & (other["vehicle_y"] < y + 60) & (
                #                               other["timestep_time"] == row['timestep_time'])]
                # other_veh['vehicle_x'] = other_veh['vehicle_x'] + np.random.normal(0, 1, len(other_veh.index))
                # other_veh['vehicle_y'] = other_veh['vehicle_y'] + np.random.normal(0, 1, len(other_veh.index))
                # D = other_veh[['vehicle_id', 'vehicle_x', 'vehicle_y', 'u', 'v', "vehicle_angle"]].to_numpy()
                D = dataset.get_nearby_vehicles(veh, row['timestep_time'])
                r = np.random.random(len(D))
                D = D[random_threshold < r, :]
                for d in D:
                    if d[0] not in ground_truth:
                        ground_truth[d[0]] = []
                    ground_truth[d[0]].append(d[1:].astype(float))
                tracker.update(D, row['timestep_time'])
                time += 1

            # fig = plt.figure(figsize=(10, 5))
            # fig.suptitle(veh)
            # ax1 = fig.add_subplot(1, 2, 1)
            # for veh, track in ground_truth.items():
            #     track = np.array(track)
            #     ax1.plot(track[:, 0], track[:, 1], alpha=0.5)
            # ax1.title.set_text("GT")
            #
            # ax2 = fig.add_subplot(1, 2, 2)
            # for track in tracker.tracks:
            #     path = np.array(track.path).astype(float)
            #     ax2.plot(path[:, 0], path[:, 1])
            # ax2.title.set_text("KF")
            #
            # plt.show()
            #
            # del fig

            model = MLP(4 + 1, 3, num_neurons=64, hidden_layers=5)
            model.load_state_dict(torch.load("Models/4_3_5_64_1000000.0.pt"))

            for track in tracker.tracks:
                for time, path, last_id, times in iter(track.get_valid_paths(4, 3)):
                    ego = veh0[veh0.timestep_time == time][['vehicle_x', 'vehicle_y', "vehicle_angle"]].to_numpy().flatten()
                    future = dataset.get_vehicle_GT(last_id, times)[['vehicle_x', 'vehicle_y', 'u', 'v', 'vehicle_angle']].to_numpy()
                    if len(future) < 7:
                        continue
                    path[-3:, :] = future[-3:, :]
                    path = object_to_ego(ego, path)
                    # path = interpolate_missing(path, 2)
                    previous = path[:5, :]
                    future = path[5:, :2]


                    gt = torch.tensor(previous)
                    gt = torch.unsqueeze(gt, 0).float().flatten(start_dim=1)
                    output = model(gt).detach().numpy()
                    output = output.squeeze(0).reshape(future[:, :2].shape)

                    ades.append(ADE_I(output, future))

    ades = np.array(ades)
    ades = np.mean(ades, axis=0)
    plt.clf()
    plt.plot(np.arange(1, len(ades) + 1, 1), ades, marker='o')
    plt.xlabel("Seconds into the Future")
    plt.ylabel("Average Distance from GT")
    plt.savefig("Average Displacement per timestep")

    print(ades)

