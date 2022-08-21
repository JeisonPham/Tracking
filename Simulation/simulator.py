import pygame
from pygame.locals import *
import pandas as pd
import numpy as np
from DotObject import DotObject
import torch
from CarObject import CarObject
from PlayerObject import PlayerObject
from MapGenerator import *
from Planning.util import objects2frame, get_corners, get_grid, render_observation, render_observations_and_traj
from PredictTrajectory import PredictTrajectory
import tools

import cv2
import matplotlib.pyplot as plt

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

    center = np.append(player._pos[:-1], [np.cos(np.deg2rad(-player.angle)), np.sin(np.deg2rad(-player.angle))])

    pts = np.hstack([pts, np.ones((pts.shape[0], 1)), np.zeros((pts.shape[0], 1))])
    objs = local2object(pts, center)
    return objs[:, :2]


def real_to_pixel(pts, player):
    pts = pts.copy()
    center = np.append(player._pos[:-1], [np.cos(np.deg2rad(-player.angle)), np.sin(np.deg2rad(-player.angle))])

    objs = object2local(pts, center)
    pts = np.round(
        (objs[:, :2] - BX[:2] + DX[:2] / 2.) / DX[:2]
    ).astype(np.int32)
    pts[:, [1, 0]] = pts[:, [0, 1]]

    return pts

    # pts = np.round(
    #     (pts - BX[:2] + DX[:2] / 2.) / DX[:2]
    # ).astype(np.int32)
    # pts[:, [1, 0]] = pts[:, [0, 1]]


def render_map(player, carList, points):
    # center = self.data[scene][name]['interp'](t0)
    center = np.append(player._pos[:-1], [np.cos(np.deg2rad(-player.angle)), np.sin(np.deg2rad(-player.angle))])
    #
    # centerlw = self.data[scene][name]['lw']
    centerlw = [1.73, 4.084][::-1]
    objs = []
    for car in carList:
        pos = car._pos.flatten()
        objs.append([pos[0], pos[1], np.cos(np.deg2rad(car.angle)), np.sin(np.deg2rad(car.angle))])
    objs = np.array(objs)

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


class Simulator:
    def __init__(self, data_file, point_cloud, start_origin=[0, 0], scale=1.0):
        """
        Simulator Object. Controls the scaling and zooming of converting between pygame and global coordinates.
        Global coordinates are any vector in R^2 so (-1, -1), (1.5, 2.5), etc
        :param data_file: file that contains information on positions of vehicles
        :param start_origin: define what  global position is at the center of the pygame
        :param scale: defines meter length. default 1px = 1meter
        """

        self._data_file = data_file
        self._point_cloud = get_map(point_cloud)[:, (0, 2)]
        self._renderMap = True

        self._start_origin = start_origin
        self._scale = scale
        self._rotation = 0

        self._FPS = pygame.time.Clock()
        self._FPS.tick(1)
        self.Frame = 0
        self._clock = pygame.time.Clock()

        self._running = True
        self._paused = False

        self._display_surf = None
        self._background_surf = None
        self._foreground_surf = None
        self.size = self.width, self.height = 640, 400

        self._arrowKeys = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]
        self._wasd = [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d]
        self._angle = [90, 270, 180, 0]
        self._movement = np.array([[0.0, 1.0], [0., -1.], [-1., 0.], [1., 0.]])

        self._mapSprite = pygame.sprite.Group()
        self._carSprite = pygame.sprite.Group()
        self._player = None

        data = pd.read_csv(data_file)
        data['ones'] = 1

        self._carList = []
        # for veh in data.vehicle_id.unique():
        #     if isinstance(veh, str) and "veh" in veh:
        #         veh0 = data[data.vehicle_id == veh]
        #         car = CarObject(veh0, stationary=False, world_starting_pos=[431.25, 726.85], simulator=self)
        #         self._carList.append(car)
        #         self._carSprite.add(car)
        # car = CarObject(None, stationary=True, world_starting_pos=[510.01333333, 612.72666667], simulator=self, angle=-90)
        # self._carList.append(car)
        # self._carSprite.add(car)

        self._MapCompleted = False

        self._SpeedController = 0

        self._start_analysis = False

        self.pred = PredictTrajectory(top_k=1000)
        self.i = 0

    @property
    def translation(self):
        corner_to_center = np.array([[1, 0, self.width / 2],
                                     [0, 1, self.height / 2],
                                     [0, 0, 1]])

        to_corner = np.array([[1, 0, -self._start_origin[0]],
                              [0, 1, -self._start_origin[1]],
                              [0, 0, 1]])
        return corner_to_center @ to_corner

    @property
    def scale(self):
        return np.array([[self._scale, 0, 0],
                         [0, self._scale, 0],
                         [0, 0, 1]])

    @property
    def zoom(self):
        first = np.array([[1, 0, -self.width / 2],
                          [0, 1, -self.height / 2],
                          [0, 0, 1]])

        second = self.scale

        third = np.array([[1, 0, self.width / 2],
                          [0, 1, self.height / 2],
                          [0, 0, 1]])
        return third @ second @ first

    @property
    def flipYAxis(self):
        return np.array([[1, 0, 0],
                         [0, 1, self.height],
                         [0, 0, 1]]) @ \
               np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 1]])

    @property
    def flipXAxis(self):
        return np.array([[1, 0, self.width],
                         [0, 1, 0],
                         [0, 0, 1]]) @ \
               np.array([[-1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])

    @property
    def rotate180(self):
        first = np.array([[1, 0, -self.width / 2],
                          [0, 1, -self.height / 2],
                          [0, 0, 1]])

        third = np.array([[1, 0, self.width / 2],
                          [0, 1, self.height / 2],
                          [0, 0, 1]])
        rotation = np.array([[-1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])

        return third @ rotation @ first

    @property
    def transformMatrix(self):
        return self.flipYAxis @ self.zoom @ self.translation

    def on_init(self):
        pygame.init()
        # for pt in self._point_cloud:
        #     self._mapSprite.add(DotObject[pt])
        self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self._foreground_surf = pygame.Surface([self.width, self.height])
        self._foreground_surf.fill((0, 0, 0))
        self._foreground_surf.set_colorkey((0, 0, 0))
        self._background_surf = pygame.Surface([self.width, self.height])

        self._running = True

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

        elif event.type == pygame.MOUSEWHEEL:
            self._scale = max(1, self._scale + event.y / 10)
            self._renderMap = True

        elif event.type == pygame.KEYDOWN:
            if event.key in self._arrowKeys:
                index = self._arrowKeys.index(event.key)
                self._start_origin += self._movement[index].astype(float) * 50 / self._scale

                self._renderMap = True

            if event.key == pygame.K_p:
                self._paused = not self._paused

            if event.key == pygame.K_f:
                self.Frame += 10

            if event.key == pygame.K_b:
                self.Frame += -10

            if event.key in self._wasd and self._player:
                index = self._wasd.index(event.key)
                self._player.update(self._angle[index])

            if event.key == pygame.K_h:
                self._start_analysis = not self._start_analysis
                self._paused = False
                print(self._start_analysis)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:

            x, y = pygame.mouse.get_pos()
            temp = np.array([x, y, 1])
            world_location = np.linalg.inv(self.transformMatrix) @ temp
            print(world_location)

            self._player = PlayerObject(world_starting_pos=world_location[:2].tolist(), simulator=self)

    def on_loop(self):
        pass

    @tools.SavePlots(save_location="F:/Radar Reseach Project/Tracking/Simulation/Output_files/")
    def on_render(self):
        if self._MapCompleted:
            # self._start_origin = self._carList[0]._pos
            self._renderMap = True
            self._foreground_surf.fill((0, 0, 0))
            self._carSprite.update()
            self._carSprite.draw(self._foreground_surf)

            if self._player:

                if self._start_analysis:
                    map = render_map(self._player, self._carList, self._point_cloud)
                    # render_observation(map)
                    # plt.show()

                    node_position = np.array([*self._player.manager.current_node.starting_position, 0, 0]).reshape(1,
                                                                                                                   -1)
                    node_pixel = real_to_pixel(node_position, self._player).flatten()
                    node_angle = np.arctan2(node_pixel[1] - 128, node_pixel[0] - 56.5)

                    traj = self.pred(map, node_pixel, node_angle)
                    traj_index = []
                    for t in traj:
                        traj_index.append(min(np.linalg.norm(t - node_pixel[::-1], axis=1)))
                    traj_index = np.argsort(traj_index)
                    traj = traj[traj_index][:10]
                    render_observations_and_traj(map, None, traj)
                    plt.scatter(node_pixel[1], node_pixel[0], s=20)

                    target = traj[0][0, (1, 0)]
                    plt.scatter(target[1], target[0], s=20, c='r', marker='x')
                    self._player.manager.plot(real_to_pixel, self._player)



                    objs = pixel_to_real(traj[0][:, (1, 0)], self._player)
                    objs = np.hstack([objs, np.ones((objs.shape[0], 1))]).T
                    self._player.update(new_pos=objs[:, 0])
                    self._player.manager.update_node(objs[:-1, 0])


                    # plt.show()

                    # self._start_analysis = False
                else:
                    self._player.update()
                self._player.draw(self._foreground_surf)

        if self._renderMap:
            self._background_surf.fill((0, 0, 0))
            ones = np.ones(self._point_cloud.shape[0]).reshape(-1, 1)
            temp = self.transformMatrix @ np.hstack([self._point_cloud, ones]).T
            # temp = self.transformMatrix @ np.array([504.68, 612.06, 1])
            temp = temp[:-1, :].T.astype(int)
            kept = np.logical_and(0 <= temp[:, 0], temp[:, 0] <= self.width)
            kept = np.logical_and(kept, 0 <= temp[:, 1])
            kept = np.logical_and(kept, temp[:, 1] <= self.height)
            temp = temp[kept]

            for pt in temp:
                pygame.draw.circle(self._background_surf, pygame.Color(255, 255, 255), pt, radius=1 * self._scale)

            self._renderMap = False
            if not self._MapCompleted:
                self._MapCompleted = True

        self._display_surf.blit(self._background_surf, (0, 0))
        self._display_surf.blit(self._foreground_surf, (0, 0))
        pygame.display.update()

    def on_cleanp(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while self._running:
            self._clock.tick(1)
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
            if self._MapCompleted and not self._paused:
                self.Frame += 1
        self.on_cleanp()


if __name__ == "__main__":
    sim = Simulator("..\Data\downtown_SD_10thru_50count_with_cad_id.csv",
                    "..\Data\downtown_SD_10_7.ply",
                    start_origin=[599, 740], scale=3.0)
    sim.on_execute()
