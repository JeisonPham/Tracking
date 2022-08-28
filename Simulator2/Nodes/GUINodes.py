from Simulator2.o3dElements import gui
import Simulator2.Nodes as Nodes
import numpy as np


# Rect (1680, 539), 240 x 539 True True
# Rect (1680, 1060), 240 x 20 True True
class PlaybackNode(Nodes.VisualNode):
    def __init__(self, computational_node=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.computational_node = computational_node
        self.info = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.info.set_on_value_changed(self._on_time_change)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(gui.Label("Time"))
        h.add_child(self.info)
        h.add_stretch()
        self.add_child(h)

        self.add_fixed(1)

        h = gui.Horiz()

        self._play = gui.Button("Play")
        self._play.set_on_clicked(self._on_play)
        self._play.horizontal_padding_em = 0.5
        self._play.vertical_padding_em = 0

        self._next = gui.Button(">")
        self._next.set_on_clicked(self._on_next)
        self._next.horizontal_padding_em = 0.5
        self._next.vertical_padding_em = 0

        self._prev = gui.Button("<")
        self._prev.set_on_clicked(self._on_prev)
        self._prev.horizontal_padding_em = 0.5
        self._prev.vertical_padding_em = 0

        h.add_stretch()
        h.add_child(self._prev)
        h.add_child(self._play)
        h.add_child(self._next)
        h.add_stretch()

        self.add_child(h)

    def on_key(self, event):
        return gui.Widget.EventCallbackResult.HANDLED

    def step(self):
        self.info.set_value(self.computational_node.time)
        self.simulator.window.set_needs_layout()

    def _on_play(self):
        self._play.text = "Stop"
        self._play.set_on_clicked(self._on_stop)
        self.simulator.start_computational_thread()

    def _on_stop(self):
        self._play.text = "Play"
        self._play.set_on_clicked(self._on_play)
        self.simulator.stop_computational_thread()

    def _on_next(self):
        self.computational_node.next()
        self.info.set_value(self.computational_node.time)
        self.simulator.window.set_needs_layout()

    def _on_prev(self):
        self.computational_node.prev()
        self.info.set_value(self.computational_node.time)
        self.simulator.window.set_needs_layout()

    def _on_time_change(self, value):
        self.computational_node.time = value
        self.computational_node.step()


class InfoNode(Nodes.VisualNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.info = gui.Label("Test")
        # self.info.visible = False
        self.add_child(self.info)

    def create_layout(self, *args, **kwargs):
        layout_context = args[0]
        frame = self.simulator.window.content_rect
        pref = self.info.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.frame = gui.Rect(frame.x, frame.get_bottom() - pref.height, pref.width, pref.height)


class CreateNode(Nodes.VisualNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vertical = gui.CollapsableVert("Create Ego/Object", 3, gui.Margins(3, 0, 3, 0))
        self.add_child(vertical)

        vertical.add_child(gui.Label("Position Vector"))
        self._pos = gui.VectorEdit()
        vertical.add_child(self._pos)

        vertical.add_child(gui.Label("Angle Vector"))
        self._ang = gui.VectorEdit()
        vertical.add_child(self._ang)

        h = gui.Horiz()
        self._ego = gui.Button("Ego")
        self._ego.set_on_clicked(self._add_ego)
        self._ego.horizontal_padding_em = 0.5
        self._ego.vertical_padding_em = 0

        self._obj = gui.Button("Obj")
        self._obj.set_on_clicked(self._add_obj)
        self._obj.horizontal_padding_em = 0.5
        self._obj.vertical_padding_em = 0

        h.add_stretch()
        h.add_child(self._ego)
        h.add_child(self._obj)
        h.add_stretch()

        vertical.add_child(h)

    def on_start(self):
        self.computational_node = self.simulator.get_nodes(Nodes.CompNode)[0]

    def on_mouse_3d(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):
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

                        self._sim.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(self.simulator.window, update_label)

            self.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _add_ego(self):
        position = np.asarray(self._pos.vector_value)
        position[1] = 3
        angle = np.asarray(self._ang.vector_value)[1]

        self.computational_node.create_ego(position, angle)

    def _add_obj(self):
        pass


class ImageNode(Nodes.VisualNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vertical = gui.CollapsableVert("Image Output", 3, gui.Margins(3, 0, 3, 0))
        self.add_child(vertical)
        self._image = gui.ImageWidget()

        vertical.add_child(self._image)

    def update_image(self, image):
        self._image.update_image(image)