from base import BaseObject
import pygame

WHITE = (255, 255, 255)


class DotObject(BaseObject):
    def __init__(self, color, *args, **kwargs):
        super(DotObject, self).__init__(*args, **kwargs)
        self.color = color
        self.update_scaling()

    def update_scaling(self, scale=1):
        self.radius = 1 * scale
        self.image = pygame.Surface([self.radius, self.radius])
        self.image.fill(WHITE)
        self.image.set_colorkey(WHITE)

        pygame.draw.circle(self.image, self.color, (self.radius / 2, self.radius / 2), self.radius)
        self.rect = self.image.get_rect()

    def update(self, simulator):
        temp = simulator.transformMatrix @ self._pos
        print(self._pos, temp)
        self.rect.x = temp[0]
        self.rect.y = temp[1]
