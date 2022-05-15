import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def generate_block(cin, cout, layer_count):
    layers = nn.ModuleList()
    layers.append(nn.Conv2d(cin, cout, padding=1, kernel_size=3, stride=1))
    for _ in range(layer_count - 1):
        layers.append(nn.Conv2d(cout, cout, padding=1, kernel_size=3, stride=1))
    return nn.Sequential(*layers)


class PlanningBackbone(nn.Module):
    def __init__(self, in_channel=5, H=256, W=256):
        super(PlanningBackbone, self).__init__()
        self.cin = in_channel

        self.block1 = nn.Sequential(generate_block(self.cin, 32, 2), nn.MaxPool2d(3, stride=2, padding=1))
        self.block2 = nn.Sequential(generate_block(32, 64, 2), nn.MaxPool2d(3, stride=2, padding=1))
        self.block3 = nn.Sequential(generate_block(64, 128, 3), nn.MaxPool2d(3, stride=2, padding=1))
        self.block4 = generate_block(128, 256, 6)
        self.block5 = generate_block(32 + 64 + 128 + 256, 256, 5)

        self.resize = nn.Upsample(size=(H // 4, W // 4), mode='bilinear')

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x1 = self.resize(x1)
        x2 = self.resize(x2)
        x3 = self.resize(x3)
        x4 = self.resize(x4)
        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        return self.block5(x5)


class CostVolume(nn.Module):
    def __init__(self, T=16, H=256, W=256):
        super(CostVolume, self).__init__()

        self.deConv1 = nn.ConvTranspose2d(256, 128, padding=1, kernel_size=3, stride=2, output_padding=1)
        self.Conv1 = nn.Conv2d(128, 128, padding=1, kernel_size=3, stride=1)

        self.deConv2 = nn.ConvTranspose2d(128, 64, padding=1, kernel_size=3, stride=2, output_padding=1)
        self.Conv2 = nn.Conv2d(64, 64, padding=1, kernel_size=3, stride=1)

        self.final = nn.Conv2d(64, T, kernel_size=1)

    def forward(self, x):
        x = self.deConv1(x)
        x = self.Conv1(x)

        x = self.deConv2(x)
        x = self.Conv2(x)

        x = self.final(x)
        return x


class PlanningModel(nn.Module):
    def __init__(self, cin, cout, H, W):
        super(PlanningModel, self).__init__()

        self.backbone = PlanningBackbone(cin, H, W)
        self.CostVolume = CostVolume(cout, H, W)

        self.clip_max = 1000

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.CostVolume(x1)

        x3 = torch.clamp(x2, min=-self.clip_max, max=self.clip_max)

        return x3


if __name__ == "__main__":
    temp = np.zeros((1, 5, 256, 256))
    bb = PlanningModel(5, 16, 256, 256)
    bb.train()

    temp = torch.tensor(temp, dtype=torch.float32)
    x = bb(temp)
    print(x.shape)

    # temp = np.zeros((1, 256, 128, 128))
    # cc = CostVolume(256, T=16)
    # cc.train()
    # temp = torch.tensor(temp, dtype=torch.float32)
    # x = cc(temp)
