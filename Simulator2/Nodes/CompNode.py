import open3d as o3d
import Simulator2.Nodes as Node
import numpy as np
import pandas as pd


class CompNode(Node.HeadlessNode):
    def __init__(self, data_file, map_file, step: float = 0.25):
        super().__init__()
        self.time = 0
        self._step = step
        self._objects = {}
        self._data = {}
        self._ego = None

        data = pd.read_csv(data_file)
        for veh in data.vehicle_id.unique():
            if isinstance(veh, str) and "veh" in veh:
                veh0 = data[data.vehicle_id == veh]
                min_time = min(veh0['timestep_time'])
                max_time = max(veh0['timestep_time'])
                self._data[min_time] = {}
                self._data[min_time][veh] = {"info": veh0,
                                             "max": max_time,
                                             "min": min_time}

        pc = o3d.io.read_point_cloud(map_file)
        points = np.asarray(pc.points)

        points = points[points[:, 0] < 390]

        min_x = min(points[:, 0])
        max_y = max(points[:, 2])
        offset = np.array([-min_x, 0, -max_y])
        points = points + offset
        points[:, 2] = -points[:, 2]
        self.points = points[points[:, 1] <= 1e-4][:, (0, 2)]

    def __del__(self):
        self._objects.clear()
        # ObjectModel.NAMES.clear()
        self._data.clear()

    def on_start(self):
        self._ego = self.simulator.get_nodes(Node.EgoNode)[0]
        self._ego.visible = False
        self._load_geometry()
        self._unload_geometry_and_step()


    def on_exit(self):
        self._objects.clear()
        del self._ego

    def next(self):
        self.time += self._step
        self._load_geometry()
        self._unload_geometry_and_step()

    def prev(self):
        self.time -= self._step
        self.time = max(self.time, 0)
        self._unload_geometry_and_step()
        self._load_geometry()

    def step(self):
        self.next()

    def create_object(self, name, data):
        self._objects[name] = Node.ObjectNode(data, name, self.scene)
        self._objects[name].on_start()

    def create_ego(self, position, angle):
        self._ego.visible = True
        self._ego.manager.set_current_node(position)
        self._ego.move_to(position)
        self._ego.rotate(angle)

    def _load_geometry(self):
        data = self._data.get(self.time, None)
        if data is not None:
            for key, value in data.items():
                if key not in self._objects:
                    self.create_object(key, value['info'])

    def _unload_geometry_and_step(self):
        if self._ego.visible:
            objects = [[x.position[0], x.position[2], np.cos(np.deg2rad(x.angle)), np.sin(np.deg2rad(x.angle))] for x in
                       self._objects.values()]
            self._ego.update(self.time, np.asarray(objects), self.points)

        unload = []
        for key, value in self._objects.items():
            if not value.update(self.time):
                unload.append(key)

        for key in unload:
            del self._objects[key]


