import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_neurons=64, hidden_layers=2):
        super(MLP, self).__init__()

        self.input_fc = nn.Linear(input_dim * 5, num_neurons)
        self.hidden_layers = nn.ModuleList()
        for x in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(num_neurons, num_neurons))
            self.hidden_layers.append(nn.Sigmoid())
            self.dropout = nn.Dropout(p=0.2)
        self.activation = nn.ReLU()
        self.output_fc = nn.Linear(num_neurons, output_dim * 2)

    def forward(self, x):
        x = self.input_fc(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.activation(x)
        return self.output_fc(x)
