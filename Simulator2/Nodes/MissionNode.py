import open3d as o3d
from Simulator2.o3dElements import gui, rendering
import Simulator2.Nodes as Node
import numpy as np
from MissionPlanner import MissionManager


class MissionNode(MissionManager, Node.VisualNode):
    def __init__(self, *args, **kwargs):
        MissionManager.__init__(self, *args, **kwargs)
        Node.VisualNode.__init__(self, register_with_panel=True)

        vertical = gui.CollapsableVert("Mission Goal", 3, gui.Margins(3, 0, 3, 0))
        self.add_child(vertical)

        self._pos = gui.VectorEdit()
        self._pos.vector_value = [0, 0, 0]
        vertical.add_child(self._pos)

        self.bubbles = []

        mat_box = rendering.MaterialRecord()
        mat_box.shader = 'defaultLitTransparency'
        # mat_box.shader = 'defaultLitSSR'
        mat_box.base_color = [0.467, 0.467, 0.467, 0.2]
        mat_box.base_roughness = 0.0
        mat_box.base_reflectance = 0.0
        mat_box.base_clearcoat = 1.0
        mat_box.thickness = 1.0
        mat_box.transmission = 0.5
        mat_box.absorption_distance = 10
        mat_box.absorption_color = [0.5, 0.5, 0.5]

        self.transparent_material = mat_box

    def on_mouse_3d(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.SHIFT):
            def depth_callback(depth_image):
                frame = self.simulator.scene.frame
                x = event.x - frame.x
                y = event.y - frame.y
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:
                    world = None
                else:
                    world = self.scene.camera.unproject(
                        event.x, event.y, depth, frame.width,
                        frame.height)

                def update_label():

                    if world is not None:
                        self._pos.vector_value = world
                        self.set_goal([world[0], world[2]])

                        self._sim.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(self.simulator.window, update_label)

            self.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def set_goal(self, goal):
        MissionManager.set_goal(self, goal)

        for bubble in self.bubbles:
            self.scene.remove_geometry(bubble)

        points = np.asarray([value.starting_position for value in self._path])

        for i, point in enumerate(points):
            name = "bubble" + str(i)
            bubble = o3d.geometry.TriangleMesh.create_sphere(self._radius)
            bubble.translate([point[0], 1, point[1]], relative=False)
            self.scene.add_geometry(name, bubble, self.transparent_material)

            self.bubbles.append(name)

    def update_node(self, vehicle):
        MissionManager.update_node(self, vehicle)
        for bubble in self.bubbles:
            self.scene.remove_geometry(bubble)

        points = np.asarray([value.starting_position for value in self._path])

        for i, point in enumerate(points):
            name = "bubble" + str(i)
            bubble = o3d.geometry.TriangleMesh.create_sphere(self._radius)
            bubble.translate([point[0], 1, point[1]], relative=False)
            self.scene.add_geometry(name, bubble, self.transparent_material)

            self.bubbles.append(name)

        bubble = o3d.geometry.TriangleMesh.create_sphere(self._radius)
        bubble.paint_uniform_color([0., 0.5, 0.5])
        bubble.translate([self.current_node.starting_position[0], 1, self.current_node.starting_position[1]],
                         relative=False)
        self.scene.add_geometry("current", bubble, self.transparent_material)
        self.bubbles.append("current")

        if len(self._path) == 0:
            self._reached_goal = True
