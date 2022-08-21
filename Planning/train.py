from comet_ml import Experiment
from comet_config import comet_config as CC
import torch
import torch.utils.data as torchdata
from tools import SimpleLoss, eval_model
from model import PlanningModel
from PlanningDataset import PlanningDataset
import numpy as np
from datetime import datetime
import logging
import os
import json

# trajectory_file = "../Planning/Trajectory_set.pkl"
# fileLocation = "../Data"
# carFile = "downtown_SD_10thru_50count_with_cad_id.csv"
# cloudFile = "downtown_SD_10_7.ply"

if not os.path.exists("log"):
    os.mkdir("log")

logging.basicConfig(filename=f"log/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log", filemode='w',
                    level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Starting Training")


def train(trajectory_file, fileLocation, carFile, cloudFile, model_location, use_resnet, hyper_params, device):
    use_resnet = use_resnet == 1
    experiment = Experiment(**CC, disabled=True)
    experiment.log_parameters(hyper_params)

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
    model.train()

    loss_fn = SimpleLoss(trajectory_file)

    opt = torch.optim.Adam(model.parameters(), lr=hyper_params["lr"], weight_decay=hyper_params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min")

    counter = 0
    with experiment.train():
        for epoch in range(int(hyper_params['epoch'])):
            print(f"Starting Epoch {epoch}")
            logging.info(f"Starting Epoch: {epoch}")
            update_lr = False
            for batchi, (x, y, gt) in enumerate(train_dataloader):
                counter += 1
                opt.zero_grad()
                pred = model(x.to(device))

                loss = loss_fn(pred, gt[1].to(device), gt[0].to(device))
                loss.backward()
                opt.step()

                if counter % 200 == 0:
                    acc, ade, fde = eval_model(test_dataloader, model, loss_fn, device)
                    experiment.log_curve(f"ADE_{counter}", x=np.arange(0, 4, 0.25), y=ade, step=counter)
                    experiment.log_metric("Eval Loss", acc, step=counter)
                    experiment.log_metric("FDE", ade[-1], step=counter)
                    update_lr = True

                if counter % 2000 == 0:
                    model.eval()
                    mname = f"{model_location}/Planner_model{counter}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pt"
                    print("saving", mname)
                    logging.info(f"Saving {mname}")
                    torch.save(model.state_dict(), mname)
                    model.train()

                if counter % 10 == 0:
                    print(epoch, batchi, counter, loss.detach().item())
                    logging.info(f"{epoch} {batchi} {counter} {loss.detach().item()}")
                    experiment.log_metric("train/loss", loss.item(), step=counter)
                    experiment.log_metric("train/epoch", epoch, step=counter)
            if update_lr:
                scheduler.step(acc)
                experiment.log_metric("train/lr", scheduler.get_lr(), step=counter)


# train(trajectory_file, fileLocation, carFile, cloudFile, 0)

if __name__ == "__main__":
    with open("config.json", "r") as file:
        params = json.load(file)

    if not os.path.exists(params['model_location']):
        os.mkdir(params['model_location'])
        print("Model Folder Created")

    train(trajectory_file=params['trajectory_file'],
          fileLocation=params['data_location'],
          carFile= params['car_file'],
          cloudFile=params['cloud_file'],
          use_resnet=params['use_resnet'],
          hyper_params=params['hyper_params'],
          model_location=params['model_location'],
          device=0)
