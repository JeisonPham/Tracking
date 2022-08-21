from Planning.tools import SimpleLoss
from Planning.model import PlanningModel
import torch
import sys
import numpy as np

sys.path.append("../Planning")


class PredictTrajectory:
    def __init__(self, top_k=3):
        self.model = PlanningModel(True, 5, 16, 256, 256)
        self.model.load_state_dict(torch.load('../Models/best_performance_resnet.pt'))
        self.model.eval()

        self.loss_fn = SimpleLoss('../Planning/Trajectory_set.pkl')
        self.top_k = top_k

    def __call__(self, x, node_position, node_angle):
        with torch.no_grad():
            x = torch.tensor(x).unsqueeze(0)

            cost_volume = self.model(x.float())
            # predicted_trajectory = self.loss_fn.top_trajectory_within_angle_to_target(cost_volume, top_k=self.top_k,
            #                                                                           target_position=node_position,
            #                                                                           target_angle=node_angle)
            predicted_trajectory = self.loss_fn.top_trajectory(cost_volume, top_k=self.top_k)

        return predicted_trajectory[0]
