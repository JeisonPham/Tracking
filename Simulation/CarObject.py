from base import BaseObject
import pygame
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class CarObject(BaseObject):
    def __init__(self, data, stationary=False, *args, **kwargs):
        super(CarObject, self).__init__(*args, **kwargs)

        self.stationary = stationary
        self.create_object()

        if not stationary:
            x = data['timestep_time'].to_numpy()
            y = data[['vehicle_angle', 'vehicle_x', 'vehicle_y', 'ones']].to_numpy()
            # pc = self.simulator._point_cloud
            # # plt.gca().invert_yaxis()
            # plt.scatter(pc[:, 0], pc[:, 1], s=1)
            # plt.plot(y[:, 1], y[:, 2], 'r')
            # plt.xlim([300, 700])
            # plt.ylim([300, 800])
            # plt.show()
            # print(y, x)
            self._maxTime = x[-1]
            self._minTime = x[0]

            self._name = data['vehicle_id'].to_numpy()[0]

            self.interp = interp1d(x, y, kind='linear', axis=0)

    def create_object(self, rotation=0, x=0, y=0):
        self.width = 1.73 * self.simulator._scale
        self.height = 4.084 * self.simulator._scale

        self.image = pygame.Surface([self.width, self.height])
        self.image.fill((255, 0, 0))
        self.image.fill((0, 255, 255), (0, 0, self.width, 2 * self.simulator._scale))
        self.image.set_colorkey((0, 0, 0))
        # pygame.draw.rect(self.image, pygame.Color(0, 255, 255), pygame.Rect(0, 0, self.width, self.height))
        self.image = pygame.transform.rotozoom(self.image, rotation, 1)
        # pygame.draw.rect(self.image, pygame.Color(0, 255, 255), pygame.Rect(0, 0, self.width, self.height))

        self.rect = self.image.get_rect(center=(0, 0))
        self.rect.center = [x, y]

    def update(self):
        if self.stationary:
            temp = self.simulator.transformMatrix @ self._pos
            self.create_object(self.angle, temp[0], temp[1])

        elif self.simulator._MapCompleted:


            frame = self.simulator.Frame
            if frame / 4 < self._minTime:
                pass
            elif frame / 4 > self._maxTime:
                self.kill()
                print(f"{self._name} has died")
            elif not self.simulator._paused:
                temp = self.interp(frame / 4)
                self.angle = np.round(temp[0])
                self._pos = temp[1:].flatten()
                temp = self.simulator.transformMatrix @ temp[1:]
                self.create_object(-self.angle, temp[0], temp[1])



