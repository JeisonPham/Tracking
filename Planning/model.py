import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
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


class FPNSect(nn.Module):
    def __init__(self, in_channel, H=256, W=256):
        super(FPNSect, self).__init__()
        self.cin = in_channel

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.cin, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 1024, 3, stride=2, padding=1)
        )

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(1024)

        self.lat4 = nn.Conv2d(512, 1024, 1, stride=2, padding=2)
        self.lat3 = nn.Conv2d(256, 512, 1, stride=2, padding=0)
        self.lat2 = nn.Conv2d(128, 256, 1, stride=2, padding=0)
        self.lat1 = nn.Conv2d(64, 128, 1, stride=2, padding=1)

        self.deconv4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, stride=2)
        )

        self.deconv3 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, stride=2)
        )

        self.deconv2 = nn.Sequential(
            nn.Conv2d(512, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, stride=2)
        )

        self.deconv1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, output_padding=1)
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # x5 = self.layer5(x4)

        # l4 = F.relu(self.bn4(self.lat4(x4)))
        # print(x5.shape, l4.shape)
        # up1 = x5 + l4

        # x4 = self.deconv4(up1)
        l3 = F.relu(self.bn3(self.lat3(x3)))
        up2 = torch.cat([x4, l3], dim=1)

        x3 = self.deconv3(up2)
        l2 = F.relu(self.bn2(self.lat2(x2)))
        # print(x3.shape, l2.shape)
        up3 = torch.cat([x3, l2], dim=1)

        x2 = self.deconv2(up3)
        l1 = F.relu(self.bn1(self.lat1(x1)))
        # print(x2.shape, l1.shape)
        up4 = torch.cat([x2, l1], dim=1)

        out = self.deconv1(up4)
        return out


class CostVolume(nn.Module):
    def __init__(self, T=16, H=256, W=256):
        super(CostVolume, self).__init__()

        self.deConv1 = nn.ConvTranspose2d(64, 128, padding=1, kernel_size=3, stride=2, output_padding=1)
        self.Conv1 = nn.Conv2d(128, 128, padding=1, kernel_size=3, stride=1)

        self.deConv2 = nn.ConvTranspose2d(128, 64, padding=1, kernel_size=3, stride=2, output_padding=1)
        self.Conv2 = nn.Conv2d(64, 64, padding=1, kernel_size=3, stride=1)

        self.final = nn.Conv2d(64, T, kernel_size=1)
        self.upsample = nn.Upsample((H, W))

    def forward(self, x):
        x = self.deConv1(x)
        x = self.Conv1(x)

        x = self.deConv2(x)
        x = self.Conv2(x)

        x = self.final(x)
        return self.upsample(x)

def gen_up(inchannels, outchannels, scale):
    return nn.Sequential(
        nn.Conv2d(inchannels, outchannels, 3, padding=1, bias=False),
        nn.BatchNorm2d(outchannels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=scale)
    )


class CNN(nn.Module):
    def __init__(self, cin, cout,
                 with_skip, dropout_p, H, W):
        super(CNN, self).__init__()

        self.cin = cin
        self.cout = cout
        self.trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.trunk.conv1 = nn.Conv2d(cin, 64, kernel_size=7,
                                     stride=2, padding=3,
                                     bias=False)
        self.downscale = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2,
                      padding=1, bias=False),
            )
        self.upscale0 = gen_up(512, 512, scale=2)
        self.upscale1 = gen_up(512, 256, scale=4)

        if with_skip:
            self.upscale2 = gen_up(256+128, 256, scale=2)
            self.upscale3 = gen_up(256+64, 128, scale=4)
        else:
            self.upscale2 = gen_up(256, 256, scale=2)
            self.upscale3 = gen_up(256, 128, scale=4)
        self.with_skip = with_skip
        self.final = nn.Conv2d(128, cout, 1)
        self.dropout = nn.Dropout(p=dropout_p, inplace=False)
        self.resize = nn.Upsample(size=(H // 4, W // 4), mode='bilinear')

    def forward(self, x):
        x = self.trunk_forward(x)
        return x

    def trunk_forward(self, x):
        x = self.trunk.conv1(x)
        x = self.trunk.bn1(x)
        x = self.trunk.relu(x)
        x = self.trunk.maxpool(x)

        x1 = self.trunk.layer1(x)
        x1 = self.dropout(x1)

        x2 = self.trunk.layer2(x1)
        x2 = self.dropout(x2)

        x = self.trunk.layer3(x2)
        x = self.dropout(x)

        x = self.trunk.layer4(x)
        x = self.downscale(x)

        x = self.upscale0(x)
        x = self.upscale1(x)

        if self.with_skip:
            x = self.upscale2(torch.cat((x, x2), 1))
            x = self.upscale3(torch.cat((x, x1), 1))
        else:
            x = self.upscale2(x)
            x = self.upscale3(x)

        x = self.final(x)

        return self.resize(x)


class PlanningModel(nn.Module):
    def __init__(self, use_resnet, cin, cout, H, W):
        super(PlanningModel, self).__init__()

        if use_resnet:
            self.backbone = CNN(cin, 256, False, 0.1, H, W)
        else:
            self.backbone = FPNSect(cin, H, W)
        self.CostVolume = CostVolume(cout, H, W)

        self.clip_max = 1000

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.CostVolume(x1)

        x3 = torch.clamp(x2, min=-self.clip_max, max=self.clip_max)

        return x3


if __name__ == "__main__":
    temp = np.zeros((1, 5, 256, 256))
    bb = PlanningModel(False, 5, 16, 256, 256)
    # bb = FPNSect(5, 256, 256)
    bb.train()

    temp = torch.tensor(temp, dtype=torch.float32)
    x = bb(temp)
    print(x.shape)

    # temp = np.zeros((1, 256, 128, 128))
    # cc = CostVolume(256, T=16)
    # cc.train()
    # temp = torch.tensor(temp, dtype=torch.float32)
    # x = cc(temp)
