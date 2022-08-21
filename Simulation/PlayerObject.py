import numpy as np
from base import BaseObject
import pygame
from Planning.model import *
import torch
from MissionPlanner import MissionManager


class PlayerObject(BaseObject):
    def __init__(self, *args, **kwargs):
        super(PlayerObject, self).__init__(*args, **kwargs)
        self.angle = 0
        self.create_object()
        self.manager = MissionManager(self._pos, radius=10, node_file="../MissionPlanner/nodes.json")
        self.manager.set_goal(np.array([601, 622.]))


    def create_object(self, rotation=0):
        self.width = 1.73 * self.simulator._scale
        self.height = 4.084 * self.simulator._scale

        self.image = pygame.Surface([self.width, self.height])
        self.image.fill((0, 255, 0))
        self.image.fill((0, 0, 255), (0, 0, self.width, 2 * self.simulator._scale))
        self.image.set_colorkey((0, 0, 0))
        self.image = pygame.transform.rotozoom(self.image, rotation, 1)
        # pygame.draw.rect(self.image, pygame.Color(0, 255, 255), pygame.Rect(0, 0, self.width, self.height))

        self.rect = self.image.get_rect(center=(0, 0))

        self.rect.center = (self.simulator.transformMatrix @ self._pos)[:2]

    def update(self, angle=None, new_pos=None):
        if angle:
            self.angle = angle
        elif new_pos is not None:
            new_pos = new_pos.reshape(-1, 1)
            temp = new_pos - self._pos
            self.angle = (np.rad2deg(np.arctan2(temp[1], temp[0])[0]) - 90) % 360
            self._pos = new_pos
            self.create_object(self.angle)
        elif self.simulator._MapCompleted:
            self.create_object(self.angle)

