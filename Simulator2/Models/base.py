import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
import Exceptions
import time


class Base:
    # NAMES = []

    def __init__(self, name: str,
                 geometry: o3d.cpu.pybind.geometry.Geometry3D,
                 material: o3d.cpu.pybind.visualization.rendering.MaterialRecord,
                 scene: rendering.Open3DScene,
                 position: np.ndarray = [0, 0, 0]):
        """
        Base class for objects to be rendered into the scene. Preferred if all objects started at [0, 0, 0] and let
        Base class handle moving it as transformation matrices default will return it to original position

        :param name: Name of the Object
        :param geometry: The Geometry of the object
        :param material: The Material
        :param scene: The rendering window that the object should render to
        :param position: Starting position of the object
        """

        # if name in Base.NAMES:
        #     raise Exceptions.DuplicateNameException(name)
        # else:
        #     Base.NAMES.append(name)

        self.name = name
        self.geometry = geometry
        self.material = material
        self.scene = scene
        self.position = position
        self.angle = 0
        self.visible = True

        self.scene.add_geometry(self.name, self.geometry, self.material)

    def __del__(self):
        if hasattr(self, "scene"):
            self.scene.remove_geometry(self.name)
            print("Destroyed")
            # Base.NAMES.remove(self.name)
            # del self.geometry
            # del self.material

    def set_visible(self, b):
        self.visible = b
        self.scene.show_geometry(self.name, b)


    @property
    def center(self):
        return self.position

    @property
    def transform(self):
        return self.scene.get_geometry_transform(self.name)

    @transform.setter
    def transform(self, trans):
        self.scene.set_geometry_transform(self.name, trans)

    def translate(self, pos: np.ndarray):
        """
        Move the Object [x, y, z] units in that specific direction
        :param pos: translation vector that specifies where to move the object
        :return: None
        """
        self.position = pos
        trans = np.eye(4)
        trans[:3, 3] = pos

        self.transform = trans @ self.transform

    def scale(self, scale: np.ndarray):
        sc = np.eye(4)
        sc[:3, :3] = scale

        self.transform = np.eye(4)
        self.transform = sc

    def move_to(self, trans: np.ndarray):
        self.position = trans
        temp = np.eye(4)
        temp[:3, 3] = trans
        self.transform = temp

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

    def step(self, frame):
        return True


