import numpy as np
from SearchAlgorithms import SearchObject, AStar
from enum import Enum
import math
import matplotlib.pyplot as plt
import re
import cv2


class MotionModel(Enum):
    STOP = 0
    RIGHT0 = 1
    RIGHT30 = 2
    RIGHT45 = 3
    RIGHT60 = 4
    UP90 = 5
    LEFT120 = 6
    LEFT135 = 7
    LEFT150 = 8
    LEFT180 = 9
    LEFT210 = 10
    LEFT225 = 11
    LEFT240 = 12
    DOWN270 = 13
    RIGHT300 = 14
    RIGHT315 = 15
    RIGHT330 = 16


motion_dict_dx_dy_cost = {
    MotionModel.RIGHT0: [np.cos(np.radians(0)), np.sin(np.radians(0))],
    MotionModel.RIGHT30: [np.cos(np.radians(30)), np.sin(np.radians(30))],
    MotionModel.RIGHT45: [np.cos(np.radians(45)), np.sin(np.radians(45))],
    MotionModel.RIGHT60: [np.cos(np.radians(60)), np.sin(np.radians(60))],
    MotionModel.UP90: [np.cos(np.radians(90)), np.sin(np.radians(90))],
    MotionModel.LEFT120: [np.cos(np.radians(120)), np.sin(np.radians(120))],
    MotionModel.LEFT135: [np.cos(np.radians(135)), np.sin(np.radians(135))],
    MotionModel.LEFT150: [np.cos(np.radians(150)), np.sin(np.radians(150))],
    MotionModel.LEFT180: [np.cos(np.radians(180)), np.sin(np.radians(180))],
    MotionModel.LEFT210: [np.cos(np.radians(210)), np.sin(np.radians(210))],
    MotionModel.LEFT225: [np.cos(np.radians(225)), np.sin(np.radians(225))],
    MotionModel.LEFT240: [np.cos(np.radians(240)), np.sin(np.radians(240))],
    MotionModel.DOWN270: [np.cos(np.radians(270)), np.sin(np.radians(270))],
    MotionModel.RIGHT300: [np.cos(np.radians(300)), np.sin(np.radians(300))],
    MotionModel.RIGHT315: [np.cos(np.radians(315)), np.sin(np.radians(315))],
    MotionModel.RIGHT330: [np.cos(np.radians(330)), np.sin(np.radians(330))],

}
# [current, prev, next]
next_motion_map = {}
for i in range(len(motion_dict_dx_dy_cost)):
    keys = list(motion_dict_dx_dy_cost.keys())
    next_motion_map[keys[i]] = [keys[i], keys[(i - 1) % len(keys)], keys[(i + 1) % len(keys)]]

motion_dict_dx_dy_cost[MotionModel.STOP] = [0, 0]

# next_motion_map = {
#     MotionModel.RIGHT0: [MotionModel.RIGHT0, MotionModel.RIGHT330, MotionModel.RIGHT30],
#     MotionModel.RIGHT30: [MotionModel.RIGHT30, MotionModel.RIGHT0, MotionModel.RIGHT45],
#     MotionModel.RIGHT45: [MotionModel.RIGHT45, MotionModel.RIGHT30, MotionModel.RIGHT60],
#     MotionModel.RIGHT60: [MotionModel.RIGHT60, MotionModel.RIGHT45, MotionModel.UP90],
#     MotionModel.UP90: [MotionModel.UP90, MotionModel.RIGHT60, MotionModel.LEFT120],
#     MotionModel.LEFT120: [MotionModel.LEFT120, MotionModel.LEFT135, MotionModel.UP90],
#     MotionModel.LEFT135: [MotionModel.LEFT135, MotionModel.LEFT120, MotionModel.LEFT150],
#     MotionModel.LEFT150: [MotionModel.LEFT150, MotionModel.LEFT135, MotionModel.LEFT180],
#     MotionModel.LEFT180: [MotionModel.LEFT180, MotionModel.LEFT150, MotionModel.LEFT210],
#     MotionModel.LEFT210: [MotionModel.LEFT210, MotionModel.LEFT180, MotionModel.LEFT225],
#     MotionModel.LEFT225: [MotionModel.LEFT225, MotionModel.LEFT210, MotionModel.LEFT240],
#     MotionModel.LEFT240: [MotionModel.LEFT240, MotionModel.LEFT225, MotionModel.DOWN270],
#     MotionModel.DOWN270: [MotionModel.DOWN270, MotionModel.LEFT240, MotionModel.RIGHT300],
#     MotionModel.RIGHT300: [MotionModel.RIGHT300, MotionModel.DOWN270, MotionModel.RIGHT315],
#     MotionModel.RIGHT315: [MotionModel.RIGHT315, MotionModel.RIGHT300, MotionModel.RIGHT330],
#     MotionModel.RIGHT330: [MotionModel.RIGHT330, MotionModel.RIGHT315, MotionModel.RIGHT0]
# }


class LocalPlannerNode(SearchObject):
    OCCUPIED_MAP = None
    SCALE = 256 / 20
    INDEX = 0

    @staticmethod
    def motion_calculator(start, goal):
        keys = list(motion_dict_dx_dy_cost.keys())
        values = [(x, y) for x, y in motion_dict_dx_dy_cost.values()]

        direction = np.clip(goal - start, -1, 1).astype(int).tolist()
        return keys[values.index((direction[0], direction[1]))]

    def plot(self, ax, color):
        position = self.transform.reverse(self.position)

        num = motion_dict_dx_dy_cost[self.current_move]

        w, l = 3, 1.5
        limits = np.array([[-w / 2, l / 2],
                           [-w / 2, -l / 2],
                           [w / 2, -l / 2],
                           [w / 2, l / 2]])
        num = np.arctan2(-num[1], num[0])
        rotation = np.array([[np.cos(num), -np.sin(num)],
                             [np.sin(num), np.cos(num)]])

        box = (np.dot(limits, rotation) + position)

        rectangle = plt.Polygon(box, closed=False, color=color)
        ax.add_patch(rectangle)

    @staticmethod
    def look_ahead(next_motion, x, y):
        num = motion_dict_dx_dy_cost[next_motion]

        w, l = 2.5 * LocalPlannerNode.SCALE, 1.5 * LocalPlannerNode.SCALE
        limits = np.array([[-w / 2, l / 2],
                           [-w / 2, -l / 2],
                           [w / 2, -l / 2],
                           [w / 2, l / 2]])
        num = np.arctan2(-num[1], num[0])
        rotation = np.array([[np.cos(num), -np.sin(num)],
                             [np.sin(num), np.cos(num)]])

        box = (np.dot(limits, rotation) + np.array([x, y])).astype(int)

        viz_map = np.zeros(LocalPlannerNode.OCCUPIED_MAP.shape)
        cv2.fillPoly(viz_map, [box[:, (1, 0)]], color=1.0)
        # plt.imshow(np.clip(LocalPlannerNode.OCCUPIED_MAP - viz_map, 0, 2), origin='lower')
        # plt.title(not np.any(LocalPlannerNode.OCCUPIED_MAP[viz_map == 1.0] == 0.0))
        # plt.savefig(f"{LocalPlannerNode.INDEX}.png")
        # plt.clf()
        LocalPlannerNode.INDEX += 1
        if np.any(LocalPlannerNode.OCCUPIED_MAP[viz_map == 1.0] == 0.0):
            return False
        return True

    def get_next_searches(self):
        for next_motion in next_motion_map[self.current_move]:
            dx, dy = motion_dict_dx_dy_cost[next_motion]
            x = np.round(self.x + dx * LocalPlannerNode.SCALE * 3).astype(int)
            y = np.round(self.y + dy * LocalPlannerNode.SCALE * 3).astype(int)

            result = LocalPlannerNode.look_ahead(next_motion, x, y)
            if 0 <= x < self.transform.nx and 0 <= y < self.transform.ny:
                if LocalPlannerNode.OCCUPIED_MAP[x, y] == 1 and result:
                    yield LocalPlannerNode(x, y, self.waypoints, self.transform, current_move=next_motion)

    def calc_heuristic(self, *args, **kwargs):
        position = self.transform.reverse(self.position)
        midpoints = [self.transform((x.position + x.projected_midpoint(position)) / 2) for x in self.waypoints]
        min_dist = float('inf')
        for i, midpoint in enumerate(midpoints):
            x, y = midpoint.astype(int)
            if LocalPlannerNode.OCCUPIED_MAP[x, y] == 0.0:
                continue

            dist = np.linalg.norm(midpoint - self.position)
            if dist < min_dist:
                min_dist = dist
        return min_dist * 100

    def calc_cost(self, goal):
        return np.linalg.norm(self.position - goal.position) / 10

    def __eq__(self, other):
        return np.linalg.norm(self.position - other.position) < LocalPlannerNode.SCALE

    def __hash__(self):
        return (self.x, self.y).__hash__()

    def __init__(self, x, y, waypoints=None, transform=None, omp=None, current_move=MotionModel.UP90):
        self.x = x
        self.y = y
        self.position = np.array([x, y])
        self.current_move = current_move
        self.waypoints = waypoints
        self.transform = transform

        if omp is not None:
            LocalPlannerNode.OCCUPIED_MAP = omp
