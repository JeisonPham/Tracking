import open3d as o3d
from Simulator2.Models import Base
from Simulator2.tools import Color
import open3d.visualization.rendering as rendering
from Planning.util import get_corners, get_grid, render_observation, render_observations_and_traj
import cv2
from Planning.tools import SimpleLoss
from Planning.model import PlanningModel
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt

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


DX, BX, (NX, NY) = get_grid([-17.0, -38.5, 60.0, 38.5], [0.3, 0.3])


def get_area(points, x, y, bounds=60):
    pts = points
    keptX = np.logical_and(pts[:, 0] <= x + bounds, x - bounds <= pts[:, 0])
    keptY = np.logical_and(pts[:, 1] <= y + bounds, y - bounds <= pts[:, 1])
    kept = np.logical_and(keptX, keptY)
    pts = pts[kept]
    return pts[:, (0, 1)]


def object2local(objs, center):
    translation_matrix = np.array([[1, 0, -center[0]],
                                   [0, 1, -center[1]],
                                   [0, 0, 1]])
    ones = np.ones((objs.shape[0], 1))
    temp = np.hstack([objs[:, :2], ones])

    temp = (translation_matrix @ temp.T)
    theta = np.arctan2(center[3], center[2])
    theta_t = -theta + np.pi / 2
    rotation_matrix = np.array([[np.cos(theta_t), -np.sin(theta_t), 0],
                                [np.sin(theta_t), np.cos(theta_t), 0],
                                [0, 0, 1]])
    temp = rotation_matrix.T @ temp
    newh = (np.arctan2(objs[:, 3], objs[:, 2]) - theta).reshape(-1, 1)
    return np.hstack([temp.T[:, :2], np.cos(newh), np.sin(newh)])


def local2object(objs, center):
    translation_matrix = np.array([[1, 0, -center[0]],
                                   [0, 1, -center[1]],
                                   [0, 0, 1]])
    ones = np.ones((objs.shape[0], 1))
    temp = np.hstack([objs[:, :2], ones])

    theta = np.arctan2(center[3], center[2])
    theta_t = -theta + np.pi / 2
    rotation_matrix = np.array([[np.cos(theta_t), -np.sin(theta_t), 0],
                                [np.sin(theta_t), np.cos(theta_t), 0],
                                [0, 0, 1]])
    temp = np.linalg.inv(rotation_matrix.T @ translation_matrix) @ temp.T
    newh = (np.arctan2(objs[:, 3], objs[:, 2]) - theta).reshape(-1, 1)
    return np.hstack([temp.T[:, :2], np.cos(newh), np.sin(newh)])


def pixel_to_real(pts, player):
    pts = pts.copy()
    pts[:, [0, 1]] = pts[:, [1, 0]]
    pts = pts * DX[:2] + BX[:2] - DX[:2] / 2

    center = player

    pts = np.hstack([pts, np.ones((pts.shape[0], 1)), np.zeros((pts.shape[0], 1))])
    objs = local2object(pts, center)
    return objs[:, :2]


def real_to_pixel(pts, player):
    pts = pts.copy()
    center = player

    objs = object2local(pts, center)
    pts = np.round(
        (objs[:, :2] - BX[:2] + DX[:2] / 2.) / DX[:2]
    ).astype(np.int32)
    pts[:, [1, 0]] = pts[:, [0, 1]]

    return pts


def render_map(player, carList, points):
    # center = self.data[scene][name]['interp'](t0)
    center = player
    #
    # centerlw = self.data[scene][name]['lw']
    centerlw = [1.73, 4.084][::-1]
    objs = np.array(carList)

    lws = np.array([centerlw.copy() for obj in objs])
    if len(objs) == 0:
        lobjs = np.zeros((0, 4))
    else:
        # lobjs = objects2frame(objs[np.newaxis, :, :], center)[0]

        lobjs = object2local(objs, center)

    obj_img = np.zeros((NX, NY))
    for box, lw in zip(lobjs, lws):
        pts = get_corners(box, lw)
        pts = np.round(
            (pts - BX[:2] + DX[:2] / 2.) / DX[:2]
        ).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]
        cv2.fillPoly(obj_img, [pts], 1.0)

    # create image of ego
    center_img = np.zeros((NX, NY))
    pts = get_corners([0.0, 0.0, 1.0, 0.0], centerlw)
    pts = np.round(
        (pts - BX[:2] + DX[:2] / 2.) / DX[:2]
    ).astype(np.int32)
    pts[:, [1, 0]] = pts[:, [0, 1]]
    print(np.mean(pts, axis=0))
    cv2.fillPoly(center_img, [pts], 1.0)

    # create image of map
    angle = -np.arctan2(center[3], center[2]) + np.pi / 2
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    area = get_area(points, center[0], center[1], 70)
    pts = np.dot(area - np.array([center[0], center[1]]), rot)
    pts = np.round(
        (pts - BX[:2] + DX[:2] / 2.) / DX[:2]
    ).astype(np.int32)
    pts[:, [0, 1]] = pts[:, [1, 0]]
    map_image = np.zeros((NX, NY))
    for pt in pts:
        cv2.circle(map_image, (pt[0], pt[1]), 1, (255, 0, 0), thickness=1)

    map_image /= 255.0

    # return np.stack([center_img, obj_img, map_image]), self.get_tgt(ltgt), center
    lane, road_div = np.zeros((NX, NY)), np.zeros((NX, NY))
    x = np.stack([map_image, lane, road_div, obj_img, center_img])
    return x


class EgoModel(Base):
    def __init__(self, position, angle, objects, points, scene):
        self.position = position
        self.angle = angle
        self.objects = objects
        self.points = points

        cube1 = o3d.geometry.TriangleMesh.create_box()
        cube1.translate([-0.5, -0.5, -0.5])
        cube1.translate([0., 0., 0.5])
        cube1.paint_uniform_color(Color.GREEN)

        cube2 = o3d.geometry.TriangleMesh.create_box()
        cube2.translate([-0.5, -0.5, -0.5])
        cube2.translate([0., 0., -0.5])
        cube2.paint_uniform_color(Color.WHITE)

        combined = o3d.geometry.TriangleMesh()
        combined += cube1 + cube2

        combined.vertices = o3d.utility.Vector3dVector(np.asarray(combined.vertices) * np.array([1.733, 1.5, 2.]))

        super().__init__(name="Ego",
                         geometry=combined,
                         material=rendering.MaterialRecord(),
                         scene=scene,
                         position=[position[1], 3, position[2]])

        self.move_to(self.position)
        self.rotate(self.angle)

    def step(self, frame):
        angle = np.deg2rad(self.angle)
        ego = np.asarray([self.position[0], self.position[2], np.cos(angle), np.sin(angle)]).reshape(-1, 1)
        objs = np.asarray([[x.position[0], x.position[2], np.cos(np.deg2rad(x.angle)), np.sin(np.deg2rad(x.angle))] for x in self.objects.values()])

        map = render_map(ego, objs, self.points)
        render_observation(map)
        plt.show()


if __name__ == "__main__":
    pbr = {1: 2}
    temp = EgoModel(None, None, pbr)
    pbr[2] = 1
    print(pbr, temp.objects)
