import open3d.visualization.gui as gui
import inspect


class Playback(gui.Horiz):
    def __init__(self, em, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.em = em

        self._play = gui.Button("Play")
        self._play.horizontal_padding_em = 0.5
        self._play.vertical_padding_em = 0

        self._next = gui.Button(">")
        self._next.horizontal_padding_em = 0.5
        self._next.vertical_padding_em = 0

        self._prev = gui.Button("<")
        self._prev.horizontal_padding_em = 0.5
        self._prev.vertical_padding_em = 0

        self.add_stretch()
        self.add_child(self._prev)
        self.add_child(self._play)
        self.add_child(self._next)
        self.add_stretch()

    def add_button_callback(self, args):
        self._prev.set_on_clicked(args[0])
        self._play.set_on_clicked(args[1])
        self._next.set_on_clicked(args[2])

    def update_play_text(self, args):
        self._play.text = args


class InsertObject(gui.Vert):
    def __init__(self, em, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_child(gui.Label("Center: "))

        h = gui.Horiz()
        self._x = gui.TextEdit()
        self._x.text_value = "X:"
        self._x.set_on_value_changed(self._format_input)
        self._y = gui.TextEdit()
        self._y.text_value = "Y:"
        self._y.set_on_value_changed(self._format_input)
        self._z = gui.TextEdit()
        self._z.text_value = "Z:"
        self._z.set_on_value_changed(self._format_input)

        h.add_stretch()
        h.add_child(self._x)
        h.add_child(self._y)
        h.add_child(self._z)
        h.add_stretch()

        self.add_child(h)

        self.add_child(gui.Label("Angle: "))
        h = gui.Horiz()
        self._ax = gui.TextEdit()
        self._ax.text_value = "AX:0"
        self._ax.set_on_value_changed(self._format_input)
        self._ay = gui.TextEdit()
        self._ay.text_value = "AY:90"
        self._ay.set_on_value_changed(self._format_input)
        self._az = gui.TextEdit()
        self._az.text_value = "AZ:0"
        self._az.set_on_value_changed(self._format_input)

        h.add_stretch()
        h.add_child(self._ax)
        h.add_child(self._ay)
        h.add_child(self._az)
        h.add_stretch()

        self.add_child(h)

        h = gui.Horiz()
        self._ego = gui.Button("Ego")
        self._ego.horizontal_padding_em = 0.5
        self._ego.vertical_padding_em = 0

        self._obj = gui.Button("Obj")
        self._obj.horizontal_padding_em = 0.5
        self._obj.vertical_padding_em = 0
        h.add_stretch()
        h.add_child(self._ego)
        h.add_child(self._obj)
        h.add_stretch()

        self.add_fixed(0.5 * em)
        self.add_child(h)

    def _format_input(self, value):
        info = value.split(":")
        if len(info) == 2:
            if not info[1].isdigit():
                info[1] = ""

        widget = getattr(self, "_" + info[0].lower())
        widget.text_value = ":".join(info)
        return widget.HANDLED

    def add_button_callback(self, args):
        self._ego.set_on_clicked(args[0])
        self._obj.set_on_clicked(args[1])

    @property
    def angle(self):
        x = self._ax.text_value[3:]
        if x == "":
            x = 0

        y = self._ay.text_value[3:]
        if y == "":
            y = 0

        z = self._az.text_value[3:]
        if z == "":
            z = 0

        return list(map(float, [x, y, z]))

    @angle.setter
    def angle(self, a):
        self._ax.text_value = f"AX:{a[0]:.2f}"
        self._ay.text_value = f"AY:{a[1]:.2f}"
        self._az.text_value = f"AZ:{a[2]:.2f}"

    @property
    def position(self):
        x = self._x.text_value[2:]
        y = self._y.text_value[2:]
        z = self._z.text_value[2:]

        return list(map(float, [x, y, z]))

    @position.setter
    def position(self, pos):
        self._x.text_value = f"X:{pos[0]:.2f}"
        self._y.text_value = f"Y:{pos[1]:.2f}"
        self._z.text_value = f"Z:{pos[2]:.2f}"


class ObjectVisibility(gui.CollapsableVert):
    def __init__(self, em, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.em = em
        self.models = gui.TreeView()
        list_grid = gui.Vert(2)
        list_grid.add_child(self.models)

        self.add_child(self.models)

    def add_tree_name(self, name, is_geometry=True):
        cell = gui.CheckableTextTreeCell(name, True, None)
        self.models.add_item(self.models.get_root_item(), cell)


class Panel(gui.Vert):

    def __init__(self, em, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.em = em
        # self.playback = Playback(em)
        # self.insert_object = InsertObject(em)
        # indented_margins = gui.Margins(em, 0, em, 0)
        # self.object_visibility = ObjectVisibility(em, "Visibility", 0, indented_margins)
        # # self.object_visibility.background_color = gui.Color.red
        #
        # v = gui.Vert()
        # v.add_fixed(em)
        # v.add_child(self.playback)
        # v.add_fixed(em)
        # v.add_child(self.insert_object)
        # v.add_fixed(em)
        # v.add_child(self.object_visibility)
        #
        # self.add_child(v)

    def setup_layout(self, window):


        frame = window.content_rect
        em = window.theme.font_size

        panel_width = 25 * em
        panel_rect = gui.Rect(frame.get_right() - panel_width, frame.y, panel_width, frame.height - frame.y)
        self.frame = panel_rect

        return panel_rect
