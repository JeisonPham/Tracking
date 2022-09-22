from comet_ml import Experiment
import torch
import torch.utils.data as torchdata
from Planning.tools import SimpleLoss, eval_model
from model import PlanningModel
from PolygonDataset import PolygonDataset
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


def train(params, device):
    use_resnet = params['train'].get('use_resnet', False)
    experiment = Experiment(**params['comet'], disabled=params['train']['disable'])
    experiment.log_parameters(params['train']['hyper_params'])

    if params['train']['dataset'] == 'PolygonDataset':
        PD = PolygonDataset(**params['train']['dataset_params'])


    device = torch.device(device)

    N = len(PD)
    train_len = int(N * 0.8)
    test_len = N - train_len

    train_set, test_set = torchdata.random_split(PD, [train_len, test_len])
    train_dataloader = torchdata.DataLoader(train_set, batch_size=params['train']['batch_size'], shuffle=True, num_workers=params['train']['num_workers'])
    test_dataloader = torchdata.DataLoader(test_set, batch_size=params['train']['batch_size'], shuffle=False, num_workers=params['train']['num_workers'])

    model = PlanningModel(use_resnet, 5, 16, 256, 256).to(device)
    model.train()

    trajectory_file = params['train']['trajectory_file']
    lr = params['train']['hyper_params']['lr']
    wd = params['train']['hyper_params']['weight_decay']
    epochs = params['train']['hyper_params']['epoch']
    model_location = params['train']['model_location']

    loss_fn = SimpleLoss(trajectory_file)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min")

    counter = 0
    with experiment.train():
        for epoch in range(int(epochs)):
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

                if counter % 2000 == 0:
                    acc, ade, fde = eval_model(test_dataloader, model, loss_fn, device)
                    experiment.log_curve(f"ADE_{counter}", x=np.arange(0, 4, 0.25), y=ade, step=counter)
                    experiment.log_metric("Eval Loss", acc, step=counter)
                    experiment.log_metric("FDE", ade[-1], step=counter)
                    update_lr = True

                if counter % 2000 == 0:
                    model.eval()
                    mname = f"{model_location}/{counter}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pt"
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
    import toml
    with open("config.toml", "r") as file:
        params = toml.load(file)
    print(params)

    if not os.path.exists(params['train']['model_location']):
        os.mkdir(params['train']['model_location'])
        print("Model Folder Created")

    train(params, device=torch.device('cuda' if params['train']['device'] != 0 and
                                                torch.cuda.is_available() else 'cpu'))
