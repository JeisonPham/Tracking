import matplotlib.patches
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib
from shapely.geometry import LineString, CAP_STYLE

DEFAULT_LANE_WIDTH = 3.2


class Junction:
    def __init__(self, *args, **kwargs):
        self.id = kwargs['id']
        self.type = kwargs['type']
        self.incLane_ids = kwargs['incLanes'].split(" ") if kwargs['incLanes'] != "" else []
        self.intLane_ids = kwargs['intLanes'].split(" ") if kwargs['intLanes'] != "" else []
        self.incLanes = list()
        self.intLanes = list()
        self.shape = None
        if "shape" in kwargs:
            coords = [[float(coord) for coord in xy.split(",")[0:2]] for xy in kwargs["shape"].split(" ")]
            self.shape = np.asarray(coords)

    def polygon(self):
        if self.shape is not None:
            return self.shape

    def plot(self, ax, **kwargs):
        if self.shape is not None:
            if 'lw' not in kwargs and "linewidth" not in kwargs:
                kwargs['lw'] = 0
            if 'color' not in kwargs:
                kwargs['color'] = 'red'
            poly = matplotlib.patches.Polygon(self.shape, True, **kwargs)
            ax.add_patch(poly)
            return poly


class Edge:
    def __init__(self, *args, **kwargs):
        self.id = kwargs['id']
        self.function = kwargs.get('function', "normal")
        self.from_junction_id = kwargs.get('from', None)
        self.to_junction_id = kwargs.get('to', None)
        self.from_junction = None
        self.to_junction = None
        self.lanes = []
        self.stop_offsets = []

    def append_lane(self, lane):
        self.lanes.append(lane)
        lane.parentEdge = self

    def get_lane(self, index):
        for lane in self.lanes:
            if lane.index == index:
                return lane
        raise IndexError("Edge contains no Lane with given index")

    def lane_count(self):
        return len(self.lanes)


class LaneMarking:
    def __init__(self, alignment, linewidth, color, dashes, purpose=None, parent=None):
        self.purpose = "" if purpose is None else purpose
        self.alignment = alignment
        self.linewidth = linewidth
        self.color = color
        self.dashes = dashes
        self.parent_lane = parent

    def polygon(self, **kwargs):
        x, y = zip(*self.alignment.coords)


class Lane:
    def __init__(self, *args, **kwargs):
        self.id = kwargs['id']
        self.index = int(kwargs['index'])
        self.width = float(kwargs.get("width", DEFAULT_LANE_WIDTH))
        coords = [[float(coord) for coord in xy.split(",")[0:2]] for xy in kwargs["shape"].split(" ")]
        self.alignment = LineString(coords)
        self.shape = self.alignment.buffer(self.width / 2, cap_style=CAP_STYLE.flat)
        if self.shape.geometryType() != "Polygon":
            self.shape = self.shape.buffer(0)
            if self.shape.geometryType() == "MultiPolygon":
                self.shape = sorted(self.shape, key=lambda x: x.area)[-1]
        self.parentEdge = None
        self.incoming_connections = []
        self.outgoing_connections = []

    def inverse_lane_index(self):
        return self.parentEdge.lane_count() - self.index - 1

    def _guess_lane_markings(self):
        markings = []
        if self.parentEdge.function == 'interal':
            return markings

        lw = 0.1 * 1
        if self.inverse_lane_index() == 0:
            leftEdge = self.alignment.parallel_offset(self.width / 2 - lw, side='left')
            color, dashes = "y", (100, 0)
            markings.append(LaneMarking(leftEdge, lw, color, dashes, purpose="center", parent=self))
        else:
            leftEdge = self.alignment.parallel_offset(self.width / 2, side="left")
            color, dashes = "w", (3, 9)
            markings.append(LaneMarking(leftEdge, lw, color, dashes, purpose="lane", parent=self))
        if self.index == 0:
            rightEdge = self.alignment.parallel_offset(self.width / 2, side='right')
            color, dashes = "w", (100, 0)
            markings.append(LaneMarking(rightEdge, lw, color, dashes, purpose="center", parent=self))

        return markings

    def polygon_lane_markings(self, **kwargs):
        for marking in self._guess_lane_markings():
            poly = marking.polygon(**kwargs)

    def polygon_shape(self, **kwargs):
        return self.shape.boundary.coords


class Connection:
    def __init__(self, *args, **kwargs):
        self.from_edge_id = kwargs['from']
        self.to_edge_id = kwargs['to']
        self.from_edge = None
        self.to_edge = None
        self.from_lane_index = int(kwargs['fromLane'])
        self.to_lane_index = int(kwargs['toLane'])
        self.from_lane = None
        self.to_lane = None
        self.via_id = kwargs.get('via', None)
        self.via_lane = None
        self.dir = kwargs['dir']
        self.state = kwargs['state']
        self.tl = kwargs.get('tl', None)
        self.linkIndex = kwargs.get("linkIndex", None)

        if 'shape' in kwargs:
            coords = [[float(coord) for coord in xy.split(",")[0:2]] for xy in kwargs["shape"].split(" ")]
            self.shape = LineString(coords)
        else:
            self.shape = None


class Net:
    def __init__(self, net_file):

        self.junction = dict()
        self.edge = dict()
        self.lane = dict()
        self.connection = list()
        self.offset = (0, 0)

        element = ET.parse(net_file).getroot()
        for node in element:
            if node.tag == 'edge':
                if node.attrib.get("functin", "") == "walkingarea":
                    continue
                edge = Edge(**node.attrib)
                for child in node:
                    if child.tag == 'lane':
                        lane = Lane(**child.attrib)
                        edge.append_lane(lane)
                self.edge[edge.id] = edge
            elif node.tag == 'junction':
                junction = Junction(**node.attrib)
                self.junction[junction.id] = junction
            elif node.tag == 'location':
                self.offset = list(map(float, node.attrib['netOffset'].split(",")))
            elif node.tag == 'connection':
                connection = Connection(**node.attrib)
                self.connection.append(connection)
        self._link_objects()

    def _link_objects(self):
        for edge in self.edge.values():
            edge.from_junction = self.junction.get(edge.from_junction_id, None)
            edge.to_junction = self.junction.get(edge.to_junction_id, None)

        for connection in self.connection:
            if connection.via_id is not None:
                connection.via_lane = self._get_lane(connection.via_id)
            connection.from_edge = self.edge.get(connection.from_edge_id, None)
            if connection.from_edge is not None:
                connection.from_lane = connection.from_edge.get_lane(connection.from_lane_index)
                connection.from_lane.outgoing_connections.append(connection)
            connection.to_edge = self.edge.get(connection.to_edge_id, None)
            if connection.to_edge is not None:
                connection.to_lane = connection.to_edge.get_lane(connection.to_lane_index)
                connection.to_lane.incoming_connections.append(connection)

        for junction in self.junction.values():
            if junction.type == 'internal':
                continue
            for i in junction.incLane_ids:
                pass

    def _get_lane(self, lane_id):
        edge_id = "_".join(lane_id.split("_")[:-1])
        lane_num = int(lane_id.split("_")[-1])
        edge = self.edge.get(edge_id, None)
        return edge.get_lane(lane_num) if edge is not None else None



if __name__ == "__main__":
    Net(r"C:\Users\Jason\Sumo\2022-09-06-15-58-13\osm.net.xml")
