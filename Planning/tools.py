import torch
import torch.nn as nn
import pickle
import random
import json
import numpy as np
from Planning.Trajectory import Trajectory


class SimpleLoss(nn.Module):
    def __init__(self, trajectory_file):
        super(SimpleLoss, self).__init__()
        with open(trajectory_file, "rb") as file:
            self.trajectory_set = pickle.load(file)

        self.ReLU = nn.ReLU()

        for key_speed, value_speed in self.trajectory_set.items():
            for key_dir, value_dir in value_speed.items():

                keys = []
                traj = []
                if len(value_dir) > 0:
                    indexes = np.asarray(range(value_dir[0].mean.shape[0])).reshape(-1, 1)
                    for key, value in value_dir.items():
                        positions = value.mean.astype(int)
                        traj.append(np.hstack([indexes, positions]))
                        keys.append(key)

                self.trajectory_set[key_speed][key_dir] = {
                    "keys": np.asarray(keys, dtype=int),
                    "traj": np.asarray(traj, dtype=int)
                }


    def ADE(self, cost_volume, gt, speed, direction):
        ades = []

        for batchii in range(cost_volume.shape[0]):
            cv = cost_volume[batchii]
            trajectory_set = self.trajectory_set[speed[batchii]][direction[batchii].item()]
            traj = trajectory_set['traj']
            cvi = cv[traj[:, :, 0], traj[:, :, 1], traj[:, :, 2]]
            min_index = torch.min(torch.sum(cvi, axis=1), axis=0)[1]

            gti = gt[batchii]

            mii = min_index.item()

            ade = np.linalg.norm(traj[mii, :, 1:] - gti[-1, :, 1:].detach().numpy(), axis=1)

            ades.append(ade)

        return np.mean(ades, axis=0)

    def top_trajectory(self, cost_volume, top_k):
        cv = cost_volume[:, self.traj[:, :, 0], self.traj[:, :, 1], self.traj[:, :, 2]]
        scores = torch.sum(cv, axis=2)
        min_index = torch.argsort(scores, axis=1)
        traj = []
        score = []
        for batchii in range(cv.shape[0]):
            index = min_index[batchii, :top_k]
            traj.append(self.traj[index.cpu(), :, 1:])
            score.append(scores[batchii, index].detach().cpu().numpy())
        return traj, np.asarray(score)

    def top_trajectory_within_angle_to_target(self, cost_volume, top_k, target_position, target_angle):
        if target_angle > 0:
            angle_range = [-15, 180]
        else:
            angle_range = [-180, 15]

        # traj_angle = self.traj[:, 5][:, 1:]
        # traj_angle = np.rad2deg(np.arctan2(traj_angle[:, 1] - 128, traj_angle[:, 0] - 56.5))
        # traj_angle = np.logical_and(traj_angle >= angle_range[0], traj_angle <= angle_range[1])

        traj_distance = []
        for traj in self.traj:
            traj_distance.append(max(np.linalg.norm(traj[:, 1:] - target_position, axis=1)))
        traj_distance = np.argsort(traj_distance)[:1000]

        # indices = np.intersect1d(np.where(traj_angle), traj_distance)
        traj_angle = self.traj[traj_distance]

        cv = cost_volume[:, traj_angle[:, :, 0], traj_angle[:, :, 1], traj_angle[:, :, 2]]
        min_index = torch.argsort(torch.sum(cv, axis=2), axis=1)
        traj = []
        for batchii in range(cv.shape[0]):
            index = min_index[batchii, :top_k]
            traj.append(traj_angle[index.cpu(), :, 1:])
        return traj

    def forward(self, cost_volume, negative_trajectory, distance):
        negative_trajectory = negative_trajectory.long()
        cv1 = cost_volume[:, negative_trajectory[:, :, :, 0], negative_trajectory[:, :, :, 1],
              negative_trajectory[:, :, :, 2]]
        cv2 = cost_volume[:, negative_trajectory[:, -1, :, 0], negative_trajectory[:, -1, :, 1],
              negative_trajectory[:, -1, :, 2]]
        cv1 = cv1[range(cv1.shape[0]), range(cv1.shape[1])]
        cv2 = cv2[range(cv2.shape[0]), range(cv2.shape[1])].unsqueeze(1)

        addition = torch.sum(torch.max(torch.sum(self.ReLU(cv2 - cv1 + distance), axis=2), axis=1)[0])
        return addition


class SimpleLossPKL(nn.Module):
    def __init__(self, mask_json, pos_weight, loss_clip, device):
        super(SimpleLoss, self).__init__()

        with open(mask_json, 'r') as reader:
            self.masks = (torch.Tensor(json.load(reader)) == 1).to(device)

        self.pos_weight = pos_weight
        self.loss_clip = loss_clip
        self.use_mask = True

    def forward(self, pred, y):
        weight = torch.ones_like(pred)
        weight[y == 1] = self.pos_weight
        new_y = y.clone()
        if self.loss_clip:
            new_y[new_y == 0] = 0.01
        if self.use_mask:
            loss = F.binary_cross_entropy_with_logits(
                pred[:, self.masks],
                new_y[:, self.masks],
                weight[:, self.masks],
                reduction='mean',
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                pred,
                new_y,
                weight,
                reduction='mean',
            )
        return loss

    def accuracy(self, pred, y):
        """pred should have already been sigmoided, y is gt
        """
        B, C, H, W = pred.shape
        new_pred = pred.clone()
        # hack to make sure argmax is inside the mask
        if self.use_mask:
            new_pred[:, self.masks] += 1.0
        K = 5
        correct = new_pred.view(B, C, H*W).topk(K, 2).indices\
            == y.view(B, C, H*W).max(2, keepdim=True).indices
        final = correct.float().sum(2).sum(0)

        return final


def eval_model(test_dataloader, model, loss_fn, device):
    model.eval()
    with torch.no_grad():
        total = 0
        ade = []
        for x, y, gt in test_dataloader:
            pred = model(x.to(device)).cpu()
            loss = loss_fn(pred, gt[1], gt[0])

            total += loss.detach().item()

            ade.append(loss_fn.ADE(pred, gt[1], gt[2], gt[3]))

            if len(ade) > 3:
                break

    model.train()
    ade = np.mean(ade, axis=0)
    return total / len(test_dataloader), ade, ade[-1]
