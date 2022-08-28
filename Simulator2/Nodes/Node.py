import open3d as o3d
from Simulator2.o3dElements import gui, rendering, geometry, device
import numpy as np
from Simulator2.tools import Color
from scipy.interpolate import interp1d


class HeadlessNode:
    def __init__(self):
        self._sim = None

    @property
    def simulator(self):
        return self._sim

    @simulator.setter
    def simulator(self, sim):
        self._sim = sim
        self.scene = sim.scene.scene

    def on_start(self):
        pass

    def on_exit(self):
        pass

    def step(self):
        pass

    def on_start_computation_thread(self):
        pass

    def on_pause_computational_thread(self):
        pass

    def on_mouse_3d(self, event):
        pass

    def on_key(self, event):
        pass

    def update(self, frame):
        return True


class VisualNode(gui.Vert, HeadlessNode):
    def __init__(self, register_with_panel=True):
        gui.Vert.__init__(self)
        HeadlessNode.__init__(self)
        self.register = register_with_panel

    def create_layout(self, *args, **kwargs):
        pass


class ModelNode(HeadlessNode):
    def __init__(self, name: str,
                 geometry: o3d.geometry,
                 material: rendering.MaterialRecord = rendering.MaterialRecord(),
                 position: np.ndarray = [0, 0, 0],
                 angle: float = 0.0):
        super().__init__()

        self.name = name
        self.geometry = geometry
        self.material = material
        self.position = position
        self.angle = angle
        self._visible = True

    def __del__(self):
        if hasattr(self, "scene"):
            self.scene.remove_geometry(self.name)

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, vis: bool):
        self.scene.show_geometry(self.name, vis)
        self._visible = vis

    @property
    def center(self):
        c = self.position
        if self.position.is_cuda:
            c = c.cpu()
        return c.numpy()

    @property
    def transform(self):
        return self.scene.get_geometry_transform(self.name)

    @transform.setter
    def transform(self, trans):
        self.scene.set_geometry_transform(self.name, trans)

    def translate(self, pos: np.ndarray):
        self.position = pos
        trans = np.eye(4)
        trans[:3, 3] = pos
        self.transform = trans @ self.transform

    def scale(self, scale: np.ndarray):
        # self.scene.remove_geometry(self.name)
        # self.geometry.vertices = o3d.utility.Vector3dVector(np.asarray(self.geometry.vertices) * scale)
        self.geometry.scale(3, [0, 0, 0])

    def move_to(self, trans: np.ndarray):
        self.position = trans
        temp = np.eye(4)
        temp[:3, 3] = trans
        self.transform = temp
        # self.scene.remove_geometry(self.name)
        # self.geometry.translate(trans, relative=False)
        # self.scene.add_geometry(self.name, self.geometry, self.material)

    def rotate(self, angle):
        previous = self.transform
        self.transform = np.eye(4)
        self.angle = angle
        angle = np.deg2rad(-angle)
        trans = np.array([[np.cos(-angle), 0, np.sin(-angle), 0],
                          [0, 1, 0, 0],
                          [-np.sin(-angle), 0, np.cos(-angle), 0],
                          [0, 0, 0, 1]])
        self.transform = previous @ trans

    def on_start(self):
        self.scene.add_geometry(self.name, self.geometry, self.material)


class ObjectNode(ModelNode):
    def __init__(self, data, name, scene=None):
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
        combined.compute_vertex_normals()

        combined = geometry.TriangleMesh.from_legacy(combined, device=device)

        mat = rendering.MaterialRecord()
        mat.shader = 'defaultLit'
        super().__init__(name=name,
                         geometry=combined,
                         material=mat)
        self.scene = scene

    def update(self, frame):
        if self._minTime <= frame <= self._maxTime:
            self.scene.show_geometry(self.name, True)
            position = self.interp(frame)
            self.move_to([position[1], 3, position[2]])
            self.rotate(position[0])
            return True
        return False

# class EgoNode(ModelNode):
#     def __init__(self, position, angle, scene=None):
#         cube1 = o3d.geometry.TriangleMesh.create_box()
#         cube1.translate([-0.5, -0.5, -0.5])
#         cube1.translate([0., 0., 0.5])
#         cube1.paint_uniform_color(Color.PURPLE)
#
#         cube2 = o3d.geometry.TriangleMesh.create_box()
#         cube2.translate([-0.5, -0.5, -0.5])
#         cube2.translate([0., 0., -0.5])
#         cube2.paint_uniform_color(Color.WHITE)
#
#         combined = o3d.geometry.TriangleMesh()
#         combined += cube1 + cube2
#
#         combined.vertices = o3d.utility.Vector3dVector(np.asarray(combined.vertices) * np.array([1.733, 1.5, 2.]))
#         combined.compute_vertex_normals()
#
#         mat = rendering.MaterialRecord()
#         mat.shader = 'defaultLit'
#
#         super().__init__(name="Ego",
#                          geometry=combined,
#                          material=mat)
#
#         self.scene = scene
#         self.move_to(position)
#         self.rotate(angle)
