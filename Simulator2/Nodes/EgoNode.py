import os
import open3d as o3d
from Simulator2.o3dElements import gui, rendering, geometry, device
import numpy as np
from Simulator2.tools import Color
from Simulator2.Nodes import ModelNode
import Simulator2.Nodes as Node
from Simulator2.tools.FrameTool import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')



class EgoNode(ModelNode):
    def __init__(self, position, angle, scene=None):
        cube1 = o3d.geometry.TriangleMesh.create_box()
        cube1.translate([-0.5, -0.5, -0.5])
        cube1.translate([0., 0., 0.5])
        cube1.paint_uniform_color(Color.PURPLE)

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

        super().__init__(name="Ego",
                         geometry=combined,
                         material=mat)

        self.scene = scene
        self.position = position
        self.angle = angle

        self.prediction = PredictTrajectory(top_k=1000)


    def on_start(self):
        super().on_start()
        self.move_to(self.position)
        self.rotate(self.angle)
        self.manager = self.simulator.get_nodes(Node.MissionNode)[0]
        self.image = self.simulator.get_nodes(Node.ImageNode)[0]
        self.visible = False

    def update(self, frame, objects, points):
        center = [self.position[0], self.position[2], np.cos(np.deg2rad(self.angle)), np.sin(np.deg2rad(self.angle))]
        map = render_map(center, objects, points)
        node_position = np.array([*self.manager.current_node.starting_position, 0, 0]).reshape(1, -1)
        node_pixel = real_to_pixel(node_position, center).flatten()
        node_angle = np.arctan2(node_pixel[1] - 128, node_pixel[0] - 56.5)
        traj = self.prediction(map, node_pixel[::-1], node_angle)
        traj_index = []
        for t in traj:
            traj_index.append(min(np.linalg.norm(t - node_pixel[::-1], axis=1)))
        traj_index = np.argsort(traj_index)
        traj = traj[traj_index][:10]
        target = traj[0][0, (1, 0)]
        objs = pixel_to_real(target, center).flatten()

        temp = (objs - self.position[[0, 2]]).flatten()

        self.move_to(np.asarray([objs[0], 3, objs[1]]))
        self.rotate((np.rad2deg(np.arctan2(temp[1], temp[0])) - 90) % 360)

        self.manager.update_node(objs)

        plt.margins(x=0, y=0)
        fig = plt.figure()
        render_observations_and_traj(map, None, traj)
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        image = geometry.Image(data)
        self.image.update_image(image)

        plt.close(fig)
