import torch
import torch.nn as nn
import pickle
import random
import numpy as np
from Planning.Trajectory import Trajectory


class SimpleLoss(nn.Module):
    def __init__(self, trajectory_file):
        super(SimpleLoss, self).__init__()
        with open(trajectory_file, "rb") as file:
            self.trajectory_set = pickle.load(file)

        self.ReLU = nn.ReLU()

        traj = []
        indexes = np.asarray(range(self.trajectory_set[0].mean.shape[0])).reshape(-1, 1)
        # keys are not guaranteed to be ordered
        keys = []
        for key, value in self.trajectory_set.items():
            positions = value.mean.astype(int)
            traj.append(np.hstack([indexes, positions]))
            keys.append(key)
        self.traj = np.asarray(traj, dtype=int)
        self.keys = np.asarray(keys, dtype=int)

    def ADE(self, cost_volume, gt):
        cv = cost_volume[:, self.traj[:, :, 0], self.traj[:, :, 1], self.traj[:, :, 2]]
        min_index = torch.min(torch.sum(cv, axis=2), axis=1)[1]

        ades = []
        for batchii in range(cv.shape[0]):
            cvi = cv[batchii]
            gti = gt[batchii]

            mii = min_index[batchii]

            ade = np.linalg.norm(self.traj[mii, :, 1:] - gti[-1, :, 1:].detach().numpy(), axis=1)

            ades.append(ade)

        return np.mean(ades, axis=0)

    def top_trajectory(self, cost_volume, top_k):
        cv = cost_volume[:, self.traj[:, :, 0], self.traj[:, :, 1], self.traj[:, :, 2]]
        min_index = torch.argsort(torch.sum(cv, axis=2), axis=1)
        traj = []
        for batchii in range(cv.shape[0]):
            index = min_index[batchii, :top_k]
            traj.append(self.traj[index.cpu(), :, 1:])
        return traj

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
        temp = None
        negative_trajectory = negative_trajectory.long()
        cv1 = cost_volume[:, negative_trajectory[:, :, :, 0], negative_trajectory[:, :, :, 1],
              negative_trajectory[:, :, :, 2]]
        cv2 = cost_volume[:, negative_trajectory[:, -1, :, 0], negative_trajectory[:, -1, :, 1],
              negative_trajectory[:, -1, :, 2]]
        cv1 = cv1[range(cv1.shape[0]), range(cv1.shape[1])]
        cv2 = cv2[range(cv2.shape[0]), range(cv2.shape[1])].unsqueeze(1)

        addition = torch.sum(torch.max(torch.sum(self.ReLU(cv2 - cv1 + distance), axis=2), axis=1)[0])
        return addition
        # for batchi in range(cost_volume.shape[0]):
        #     cv = cost_volume[batchi]
        #     neg_traj = negative_trajectory[batchi].long()
        #     dist = distance[batchi]
        #
        #     gt_cv = cv[neg_traj[-1, :, 0], neg_traj[-1, :, 1], neg_traj[-1, :, 2]]
        #     neg_cv = cv[neg_traj[:, :, 0], neg_traj[:, :, 1], neg_traj[:, :, 2]]
        #
        #     t = torch.max(torch.sum(self.ReLU(gt_cv - neg_cv + dist), axis=1))
        #     if addition[0][batchi].item() == t.item():
        #         continue
        #     if temp is None:
        #         temp = t
        #     else:
        #         temp += t
        # return temp
        # total = 0
        # for i, index in enumerate(class_index.detach().numpy()):
        #     gt = self.trajectory_set[index].mean
        #     idxes = list(range(len(self.trajectory_set)))
        #     idxes.remove(index)
        #
        #     negative_idxes = random.sample(idxes, N)
        #     negative_trajectory = [self.trajectory_set[index].mean for index in negative_idxes]
        #     negative_trajectory.append(gt)
        #
        #     negative_trajectory = np.asarray(negative_trajectory)
        #
        #     max_neg = 0
        #     for neg_traj in negative_trajectory:
        #         distance = np.linalg.norm(neg_traj - gt, axis=1)
        #         gt_cost = cost_volume[i, range(16), gt[:, 0], gt[:, 1]].detach().numpy()
        #         neg_cost = cost_volume[i, range(16), neg_traj[:, 0], neg_traj[:, 1]].detach().numpy()
        #
        #         total = self.ReLU(torch.Tensor(gt_cost - neg_cost + distance))
        #         max_neg = max(max_neg, np.sum(total))
        #     total += max_neg
        # return torch.Tensor(total)


def eval_model(test_dataloader, model, loss_fn, device):
    model.eval()
    with torch.no_grad():
        total = 0
        ade = []
        for x, y, gt in test_dataloader:
            pred = model(x.to(device)).cpu()
            loss = loss_fn(pred, gt[1], gt[0])

            total += loss.detach().item()

            ade.append(loss_fn.ADE(pred, gt[1]))

            if len(ade) > 3:
                break

    model.train()
    ade = np.mean(ade, axis=0)
    return total / len(test_dataloader), ade, ade[-1]
