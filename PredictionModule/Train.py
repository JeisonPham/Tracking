import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from comet_ml import Experiment
from comet_config import comet_config as CC
from metrics import *


def train(device, train_dataset, test_dataset, model, hyper_params=None):
    experiment = Experiment(**CC)
    experiment.log_parameters(hyper_params)

    lr, epochs = hyper_params['learning_rate'], hyper_params['num_epochs']
    model_name = hyper_params['model_name']

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    min_ade = np.nan
    with experiment.train():
        for epoch in range(1, int(epochs) + 1):
            batch_loss = 0
            total = len(train_dataset)
            for i, (x, y) in enumerate(train_dataset):
                x = x.float().to(device)
                y = y.float().to(device)
                optimizer.zero_grad()
                output = model(x.flatten(start_dim=1))
                loss = criterion(output, y.flatten(start_dim=1))
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

            if epoch % 500 == 0:
                ade, fde = evaluate_model(device, epoch, test_dataset, model)
                if np.isnan(min_ade) or ade < min_ade:
                    min_ade = ade
                    torch.save(model.state_dict(), os.path.join("Models", f"{model_name}_{epochs}.pt"))
                experiment.log_metric("ADE_K", ade, step=epoch)
                experiment.log_metric("FDE", fde, step=epoch)
            experiment.log_metric("loss", batch_loss / total, step=epoch)


def evaluate_model(device, epoch, test_dataset, model):
    ade = []
    fde = []
    length = 0
    with torch.no_grad():
        for index, (x, y) in enumerate(test_dataset):
            x = x.float().to(device)
            y = y.float().detach().numpy()
            output = model(x.flatten(start_dim=1)).detach().cpu().numpy().reshape(y.shape)
            for predicted, actual in zip(output, y):
                length += 1
                ade.append(ADE_K(predicted, actual, 6))
                fde.append(FDE(predicted, actual))

    return sum(ade) / length, sum(fde) / length


