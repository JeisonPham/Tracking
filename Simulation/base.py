import pygame
from pygame.locals import *
import numpy as np


class BaseObject(pygame.sprite.Sprite):
    def __init__(self, world_starting_pos: np.ndarray, simulator: object, angle: float = 0):
        super().__init__()
        self._pos = np.array(world_starting_pos + [1]).reshape(3, 1)
        self.simulator = simulator
        self.angle = angle

    def update(self, time):
        pass

    def draw(self, surface):
        surface.blit(self.image, self.rect)
