import open3d as o3d

try:
    from base import Base
except:
    from .base import Base

import numpy as np
from tools import Color
import open3d.visualization.rendering as rendering
from scipy.interpolate import interp1d


class ObjectModel(Base):
    def __init__(self, name, data, frame, scene):
        x = data['timestep_time'].to_numpy()
        y = data[['vehicle_angle', 'vehicle_x', 'vehicle_y']].to_numpy()
        self._maxTime = x[-1]
        self._minTime = x[0]

        self._name = data['vehicle_id'].to_numpy()[0]

        self.interp = interp1d(x, y, kind='linear', axis=0)

        cube1 = o3d.geometry.TriangleMesh.create_box()
        cube1.translate([-0.5, -0.5, -0.5])
        cube1.translate([0., 0., 0.5])
        cube1.paint_uniform_color(Color.BLUE)

        cube2 = o3d.geometry.TriangleMesh.create_box()
        cube2.translate([-0.5, -0.5, -0.5])
        cube2.translate([0., 0., -0.5])
        cube2.paint_uniform_color(Color.WHITE)

        combined = o3d.geometry.TriangleMesh()
        combined += cube1 + cube2

        combined.vertices = o3d.utility.Vector3dVector(np.asarray(combined.vertices) * np.array([1.733, 1.5, 2.]))

        position = self.interp(frame)
        super().__init__(name=name,
                         geometry=combined,
                         material=rendering.MaterialRecord(),
                         scene=scene,
                         position=[position[1], 3, position[2]])

        self.step(frame)

    def step(self, frame):
        if self._minTime <= frame <= self._maxTime:
            self.scene.show_geometry(self.name, True)
            position = self.interp(frame)
            self.move_to([position[1], 3, position[2]])
            self.rotate(position[0])
            return True
        return False


if __name__ == "__main__":
    ObjectModel(None, None)
