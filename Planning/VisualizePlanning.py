import numpy as np
from PlanningDataset import *
import torch.utils.data as torchdata
from model import *
from tools import *
from util import *


def visualize_test(trajectory_file, fileLocation, carFile, cloudFile, model_location, use_resnet, hyper_params, device):
    use_resnet = use_resnet == 1


    PD = PlanningDataset(trajectory_file=trajectory_file,
                         fileLocation=fileLocation,
                         carFile=carFile,
                         cloudFile=cloudFile, ego_only=False, flip_aug=False, is_train=False,
                         t_spacing=0.25, only_y=False)

    device = torch.device(device)

    N = len(PD)
    train_len = int(N * 0.8)
    test_len = N - train_len

    train_set, test_set = torchdata.random_split(PD, [train_len, test_len])
    train_dataloader = torchdata.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=10)
    test_dataloader = torchdata.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=10)

    model = PlanningModel(use_resnet, 5, 16, 256, 256).to(device)
    model.load_state_dict(torch.load(model_location))
    model.eval()

    loss_fn = SimpleLoss(trajectory_file)

    plt.figure(figsize=(4, 4))
    counter = 0
    with torch.no_grad():
        for batchi, (x, y, gt) in enumerate(test_dataloader):
            cost_volume = model(x.to(device))
            predicted_trajectory = loss_fn.top_trajectory(cost_volume, top_k=3)
            for xi, yi, pred_traj in zip(x, y, predicted_trajectory):
                render_observations_and_traj(xi, yi, pred_traj)
                plt.savefig(f"output/visual_{counter:05d}.png", bbox_inches='tight')
                plt.clf()
                counter += 1


if __name__ == "__main__":
    with open("config.json", "r") as file:
        params = json.load(file)

    visualize_test(trajectory_file=params['trajectory_file'],
          fileLocation=params['data_location'],
          carFile= params['car_file'],
          cloudFile=params['cloud_file'],
          use_resnet=params['use_resnet'],
          hyper_params=params['hyper_params'],
          model_location="../Models/best_performance_resnet.pt",
          device=0)



