import numpy as np
from SearchAlgorithms import SearchObject, AStar
from enum import Enum
import math
import matplotlib.pyplot as plt


class MotionModel(Enum):
    """
    o-o-o-o-o
    o-------o
    o---X---o
    o-------o
    o-o-o-o-o
    """
    #
    STOP = 0
    #
    STRAIGHT = 1
    #
    LEFTFORWARD30 = 2
    LEFTFORWARD45 = 3
    LEFTFORWARD60 = 4
    #
    LEFT = 5
    #
    LEFTBACKWARD30 = 6
    LEFTBACKWARD45 = 7
    LEFTBACKWARD60 = 8
    #
    RIGHTFORWARD30 = 9
    RIGHTFORWARD45 = 10
    RIGHTFORWARD60 = 11
    #
    RIGHT = 12
    #
    RIGHTBACKWARD30 = 13
    RIGHTBACKWARD45 = 14
    RIGHTBACKWARD60 = 15


class LocalPlannerNode(SearchObject):
    def plot(self, ax):
        rectangle = plt.Rectangle((self.x - 0.5, self.y - 0.5), 1, 1, fc='blue', ec="red")
        ax.add_patch(rectangle)

    def get_next_searches(self):
        for next_motion in self.next_motion_map[self.current_move]:
            dx, dy, cost = self.motion_dict_dx_dy_cost[next_motion]
            yield LocalPlannerNode(self.x + dx, self.y + dy, next_motion)

    def calc_heuristic(self, *args, **kwargs):
        return 0

    def calc_cost(self, goal):
        return np.linalg.norm(self.position - goal.position)

    def __eq__(self, other):
        return np.linalg.norm(self.position - other.position) < 1

    def __hash__(self):
        return (self.x, self.y).__hash__()

    def __init__(self, x, y, current_move=MotionModel.LEFTFORWARD30):
        self.x = x
        self.y = y
        self.position = np.array([x, y])
        self.current_move = current_move

        self.motion_dict_dx_dy_cost = {
            MotionModel.RIGHT: [-2, 0, 2],
            MotionModel.RIGHTFORWARD60: [-2, -1, math.sqrt(5)],
            MotionModel.RIGHTFORWARD45: [-2, -2, math.sqrt(8)],
            MotionModel.RIGHTFORWARD30: [-1, -2, math.sqrt(5)],
            MotionModel.STRAIGHT: [0, -2, 2],
            MotionModel.LEFTFORWARD30: [1, -2, math.sqrt(5)],
            MotionModel.LEFTFORWARD45: [2, -2, math.sqrt(8)],
            MotionModel.LEFTFORWARD60: [2, -1, math.sqrt(5)],
            MotionModel.LEFT: [2, 0, 2],
        }
        self.next_motion_map = {
            MotionModel.RIGHT: [MotionModel.RIGHT, MotionModel.RIGHTFORWARD30],
            MotionModel.RIGHTFORWARD60: [MotionModel.RIGHTFORWARD60, MotionModel.RIGHT, MotionModel.RIGHTFORWARD45],
            MotionModel.RIGHTFORWARD45: [MotionModel.RIGHTFORWARD45, MotionModel.RIGHTFORWARD60,
                                         MotionModel.RIGHTFORWARD30],
            MotionModel.RIGHTFORWARD30: [MotionModel.RIGHTFORWARD30, MotionModel.RIGHTFORWARD45, MotionModel.STRAIGHT],
            MotionModel.STRAIGHT: [MotionModel.STRAIGHT, MotionModel.LEFTFORWARD30, MotionModel.RIGHTFORWARD30],
            MotionModel.LEFTFORWARD30: [MotionModel.LEFTFORWARD30, MotionModel.STRAIGHT, MotionModel.LEFTFORWARD45],
            MotionModel.LEFTFORWARD45: [MotionModel.LEFTFORWARD45, MotionModel.LEFTFORWARD30,
                                        MotionModel.LEFTFORWARD60],
            MotionModel.LEFTFORWARD60: [MotionModel.LEFTFORWARD60, MotionModel.LEFTFORWARD45, MotionModel.LEFT],
            MotionModel.LEFT: [MotionModel.LEFT, MotionModel.LEFTFORWARD60],
            MotionModel.STOP: [MotionModel.LEFT,
                               MotionModel.LEFTFORWARD60, MotionModel.LEFTFORWARD45, MotionModel.LEFTFORWARD30,
                               MotionModel.STRAIGHT,
                               MotionModel.RIGHTFORWARD30, MotionModel.RIGHTFORWARD45, MotionModel.RIGHTFORWARD60,
                               MotionModel.RIGHT],
        }
