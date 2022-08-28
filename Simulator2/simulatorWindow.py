import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import Models
import time
import threading
import pandas as pd
import Layout


class SimulatorWindow:
    def __init__(self, title: str, width, height, data_file: str):
        self.title = title
        self.width = width
        self.height = height
        self.frame = 0

        self._objects = {}
        self._last_animation_time = time.time()
        self._animation_delay_secs = 1
        self._step = 0.25
        self._frame_finished = True
        self._computation_thread = threading.Thread(target=self._on_start_animation)
        self._state = threading.Condition()
        self._paused = False

        self._ego = None

        self._data = {}
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

    def __del__(self):
        print(len(threading.enumerate()))
        gui.Application.instance.quit()
        for thread in threading.enumerate():
            thread.join()

    def _on_layout(self, layout_context):

        frame = self.window.content_rect
        em = self.window.theme.font_size

        pref = self.info.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.info.frame = gui.Rect(frame.x, frame.get_bottom() - pref.height, pref.width, pref.height)

        self.frame_info.frame = gui.Rect(0, 0, 3 * em, 1 * em)

        panel_rect = self._panel.setup_layout(self.window)
        self._3d.frame = gui.Rect(frame.x, frame.y, panel_rect.x - frame.x, frame.height - frame.y)

    def _on_mouse_3d(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.CTRL):

            def depth_callback(depth_image):
                x = event.x - self._3d.frame.x
                y = event.y - self._3d.frame.y
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:
                    text = ""
                    world = None
                else:
                    world = self._3d.scene.camera.unproject(
                        event.x, event.y, depth, self._3d.frame.width,
                        self._3d.frame.height)
                    text = "({:.3f}, {:.3f}, {:.3f})".format(
                        world[0], world[1], world[2])

                def update_label():
                    self.info.text = text
                    self.info.visible = (text != "")

                    if world is not None:
                        self._panel.insert_object.position = world

                    self.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(
                    self.window, update_label)

            self._3d.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _init_user_interface(self, title, width, height):
        self.window = gui.Application.instance.create_window(title, width, height)
        self.window.set_on_layout(self._on_layout)

        em = self.window.theme.font_size

        self._3d = gui.SceneWidget()
        # self._3d.enable_scene_caching(True)
        self._3d.scene = rendering.Open3DScene(self.window.renderer)
        self._3d.scene.set_background([1, 0, 0, 1])
        self._3d.scene.show_ground_plane(True, rendering.Scene.GroundPlane.XZ)
        self.window.add_child(self._3d)

        self.info = gui.Label("")
        self.info.visible = False
        self.window.add_child(self.info)

        self.frame_info = gui.Label(str(self.frame))
        self.window.add_child(self.frame_info)

        # self._panel = gui.Vert()
        self._panel = Layout.Panel(em)
        self._panel.insert_object.add_button_callback([self._add_ego, self._add_obj])
        self._panel.playback.add_button_callback([self._on_prev, self._start_computation_thread, self._on_next])
        self.window.add_child(self._panel)

        self._3d.set_on_mouse(self._on_mouse_3d)

    def setup_camera(self):
        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10],
                                                   [10, 10, 10])
        self._3d.setup_camera(120, bbox, [0, 0, 0])

    def _add_ego(self):
        position = self._panel.insert_object.position
        angle = self._panel.insert_object.angle

        if self._ego is None:
            self._ego = Models.EgoModel(position, angle[1], self._objects,
                                        self.points,
                                        self._3d.scene)
        else:
            self._ego.move_to(position + np.array([0, 1.5, 0]))
            self._ego.rotate(angle[1])

    def _add_obj(self):
        pass

    def _on_next(self):
        self.frame += self._step
        self.frame_info.text = str(self.frame)
        self._check_if_step_unload()
        self._check_if_load_geometry()

        self.window.set_needs_layout()

    def _on_prev(self):
        self.frame = max(self.frame - self._step, 0)
        self.frame_info.text = str(self.frame)
        self._check_if_step_unload()
        self._check_if_load_geometry()

        self.window.set_needs_layout()

    def _load_map(self):
        print("Loading Map...")

        def load_thread():
            # self._objects["Map"] = Models.MapModel("F:\Radar Reseach Project\Tracking\Data\downtown_SD_10_7.ply", l=2,
            #                                        scene=self._3d.scene)
            self.points = Models.MapModel.read_pc("F:\Radar Reseach Project\Tracking\Data\downtown_SD_10_7.ply")
            self._map = Models.MapModel("Models/MapFiles", l=1, scene=self._3d.scene)
            self._3d.look_at(self._map.center, self._map.center + np.array([0, 20, 0]), [0, 1, 0])
            self._3d.force_redraw()

        threading.Thread(target=load_thread).start()

    def _on_start_animation(self):
        self._panel.playback.add_button_callback([self._on_prev, self._on_stop_animation, self._on_next])
        self._panel.playback.update_play_text("Stop")
        self._last_animation_time = 0.0

        while True:
            now = time.time()
            if now >= self._last_animation_time + self._animation_delay_secs:
                self._last_animation_time = now

                def display():
                    gui.Application.instance.run_one_tick()
                    self.frame += self._step
                    self.frame_info.text = str(self.frame)
                    self._check_if_step_unload()
                    self._check_if_load_geometry()

                gui.Application.instance.post_to_main_thread(self.window, display)

            with self._state:
                if self._paused:
                    self._state.wait()

    def _start_computation_thread(self):
        self._computation_thread.start()

    def _resume_computation_thread(self):
        self._panel.playback.add_button_callback([self._on_prev, self._on_stop_animation, self._on_next])
        self._panel.playback.update_play_text("Stop")
        self._last_animation_time = 0.0

        with self._state:
            self._paused = False
            self._state.notify()

    def _on_stop_animation(self):
        self._panel.playback.update_play_text("Play")
        self._panel.playback.add_button_callback([self._on_prev, self._resume_computation_thread, self._on_next])
        with self._state:
            self._paused = True

    def _check_if_load_geometry(self):
        data = self._data.get(self.frame, None)
        if data is not None:
            def load_thread():
                for key, value in data.items():
                    if value['min'] <= self.frame <= value['max']:
                        if key not in self._objects:
                            self._objects[key] = Models.ObjectModel(key, value['info'], self.frame, self._3d.scene)
                            # self._panel.object_visibility.add_tree_name(key)
                        elif not self._objects[key].visible:
                            self._objects[key].set_visible(True)

            load_thread()

        # for key in loaded:
        #     del self._data[key]

    def _check_if_step_unload(self):

        def load_thread():
            unload = []
            for key, value in self._objects.items():
                if self._objects[key].visible:
                    if not value.step(self.frame):
                        # self._objects[key].set_visible(False)
                        unload.append(key)
            for key in unload:
                del self._objects[key]

        # threading.Thread(target=load_thread).start()
        load_thread()

        if self._ego is not None:
            self._ego.step(self.frame)

        # threading.Thread(target=self._3d.force_redraw).start()
        # self.window.post_redraw()

    def _visualize(self):
        gui.Application.instance.initialize()
        self._init_user_interface(self.title, self.width, self.height)
        # self._load_map()
        # self._check_if_load_geometry()
        # self._load_geometries()

        gui.Application.instance.run()

    def visualize(self):
        self._visualize()
        self.setup_camera()


def main():
    s = SimulatorWindow("Open3D", 1024, 768,
                        "F:\Radar Reseach Project\Tracking\Data\downtown_SD_10thru_50count_with_cad_id.csv")
    s.visualize()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
