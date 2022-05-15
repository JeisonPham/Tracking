import torch
from torch.utils.data import DataLoader, random_split
from PredictionModule.TrackingDataset import TrackingLoader
from PredictionModule.Train import train
from PredictionModule.model import MLP
import os

device = torch.device(0)


def run_experiment(data_csv, past_steps, future_steps, num_layers, num_neurons):
    data = TrackingLoader(data_csv, past_steps, future_steps, step=1,
                          max_tracks=100)
    train_size = int(0.8 * len(data))
    train_dataset, test_dataset = random_split(data, [train_size, len(data) - train_size])
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    past = int(past_steps / 1) + 1
    future = int(future_steps / 1)
    model = MLP(past, future, num_neurons=num_neurons, hidden_layers=num_layers).to(device)
    hyper_params = {
        "past": past_steps,
        "future": future_steps,
        "step": 1,
        'layers': num_layers,
        'neurons': num_neurons,
        'learning_rate': 1e-4,
        'model_name': f"{past_steps}_{future_steps}_{num_layers}_{num_neurons}",
        'num_epochs': 1e6
    }
    train(device, train_dataloader, test_dataloader, model, hyper_params)
    # plt.title(f"Past Steps: {past_steps}, Future Steps: {future_steps}, Neurons: {num_neurons}, Layers: {num_layers}, final loss: {loss[-1]}")
    # plt.plot(range(len(loss)), loss)
    # plt.savefig(f"Loss Graphs/{past_steps}_{future_steps}_{num_layers}_{num_neurons}_{loss[-1]}.png")
    # plt.clf()


def train(data_csv, past, future, layers, neurons):
    if not os.path.exists("Loss Graphs"):
        os.mkdir("Loss Graphs")

    if not os.path.exists("Models"):
        os.mkdir("Models")
    run_experiment(data_csv, past, future, layers, neurons)
