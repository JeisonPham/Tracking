import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import matplotlib
from scipy.interpolate import interp1d
from shapely.geometry import Polygon
import json

RADIUS = 10
NUM_LANE_NODES = 10


class Node:
    def __init__(self, *args, **kwargs):
        self.id = kwargs["id"]
        self.type = kwargs["type"]
        self.incLane_ids = kwargs["incLanes"].split(" ") if kwargs["incLanes"] != "" else []
        self.intLane_ids = kwargs["intLanes"].split(" ") if kwargs["intLanes"] != "" else []
        self.incLanes = []
        self.intLanes = []
        self.x = float(kwargs['x'])
        self.y = float(kwargs['y'])
        self.incoming_edges = []
        self.outgoing_edges = []

    @property
    def position(self):
        return np.asarray([self.x, self.y])

    def __repr__(self):
        return self.id

    def draw(self, position):
        position = np.array(position)
        if hasattr(self, "polygon"):
            if any(np.linalg.norm(np.array([*self.polygon.exterior.xy]).T - position, axis=1) <= 60):
                plt.plot(*self.polygon.exterior.xy)

    def get_neighbor_nodes(self):
        to = []
        for edge in self.outgoing_edges:
            to.append(edge.to)
        return to

    def plot(self):
        style = "Simple, tail_width=0.5, head_width=4, head_length=8"

        line = plt.scatter(self.position[0], self.position[1], s=RADIUS)
        kw = dict(arrowstyle=style, color="k")
        for node in self.get_neighbor_nodes():
            a = matplotlib.patches.FancyArrowPatch(self.position, node.position,
                                                   connectionstyle='arc3,rad=0.1', **kw)
            plt.gca().add_patch(a)


class Edge:
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.lanes = {}

    def __repr__(self):
        return self.id

    def add_lane(self, lane):
        self.lanes[lane.index] = lane

    def get_lane(self, index):
        if index in self.lanes:
            return self.lanes[index]
        raise IndexError

    def draw(self, position):
        position = np.array(position)
        if hasattr(self, "polygon"):
            if any(np.linalg.norm(np.array([*self.polygon.exterior.xy]).T - position, axis=1) <= 60):
                plt.plot(*self.polygon.exterior.xy)

    def link_lane_nodes(self):
        lanes = list(self.lanes.values())

        for i in range(len(lanes)):

            if i == len(lanes) - 1:
                lane1 = lanes[i]
                lane2 = lanes[i - 1]
            else:
                lane1 = lanes[i]
                lane2 = lanes[i + 1]

            # Handle Horizontal Connections
            for node_index in range(len(lane1.lane_nodes)):
                lane1.lane_nodes[node_index].link_two_way(lane2.lane_nodes[node_index])
                if node_index < len(lane1.lane_nodes) - 1:
                    lane1.lane_nodes[node_index].link_one_way(lane2.lane_nodes[node_index + 1])
                    lane1.lane_nodes[node_index].link_one_way(lane1.lane_nodes[node_index + 1])

class LaneNodes:
    EXISTS = []

    def __init__(self, position):
        self.position = position
        self.neighbors = set()

        LaneNodes.EXISTS.append(self)

    def link_two_way(self, other_laneNode):
        if self != other_laneNode:
            self.neighbors.add(other_laneNode)
            other_laneNode.neighbors.add(self)

    def link_one_way(self, other_laneNode):
        if self != other_laneNode:
            self.neighbors.add(other_laneNode)

    def draw(self):
        style = "Simple, tail_width=0.5, head_width=1, head_length=1"

        line = plt.scatter(self.position[0], self.position[1], s=RADIUS)
        kw = dict(arrowstyle=style, color="k")
        for node in self.neighbors:
            combined = np.array([self.position, node.position]).reshape(-1, 2)
            plt.plot(combined[:, 0], combined[:, 1])


class Lane:
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.parent_edge = None
        self.index = int(self.index)
        self.incoming_connection = []
        self.outgoing_connection = []

        self.shape = [x.split(",") for x in self.shape.split(" ")]
        self.shape = np.array([[float(x), float(y)] for x, y in self.shape])
        self.end = self.shape[-1]
        self.start = self.shape[0]

        interp = interp1d(x=np.linspace(0, NUM_LANE_NODES, len(self.shape)), y=self.shape, axis=0)
        self.lane_nodes = [LaneNodes(position) for position in interp(np.arange(0, NUM_LANE_NODES))]

    def __repr__(self):
        return self.id

    def draw(self, position, hide_nodes=False):
        position = np.array(position)
        if hasattr(self, "polygon"):
            if any(np.linalg.norm(np.array([*self.polygon.exterior.xy]).T - position, axis=1) <= 60):
                plt.plot(*self.polygon.exterior.xy)
                if not hide_nodes:
                    for lane_node in self.lane_nodes:
                        lane_node.draw()


class Connection:
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.fromLane = int(self.fromLane)
        self.toLane = int(self.toLane)

    def link_lane_nodes(self):
        if hasattr(self, 'via_lane'):
            self.from_lane.lane_nodes[-1].link_one_way(self.via_lane.lane_nodes[0])
            self.via_lane.lane_nodes[-1].link_one_way(self.to_lane.lane_nodes[0])

    def draw(self, position, hide_nodes=False):
        if hasattr(self, 'via_lane'):
            if not hide_nodes and (np.linalg.norm(self.via_lane.lane_nodes[0].position - position) <= 60 \
                                   or np.linalg.norm(self.via_lane.lane_nodes[-1].position - position) <= 60):
                for lane_node in self.via_lane.lane_nodes:
                    lane_node.draw()


class Net:
    def __init__(self, net_file):
        elements = ET.parse(net_file).getroot()
        self.nodes = dict()
        self.edges = dict()
        self.connections = list()
        self.lanes = dict()
        for obj in elements:
            if obj.tag == 'junction':
                if 'x' not in obj.attrib: continue
                node = Node(**obj.attrib)
                self.nodes[node.id] = node
            elif obj.tag == 'edge':
                edge = Edge(**obj.attrib)
                for child in obj:
                    if child.tag == 'lane':
                        lane = Lane(**child.attrib)
                        lane.parent_edge = edge
                        self.lanes[lane.id] = lane
                        edge.add_lane(lane)
                self.edges[edge.id] = edge
            elif obj.tag == 'connection':
                connection = Connection(**obj.attrib)
                self.connections.append(connection)

    # link up after pickling
    def link_objects(self):
        for edge in self.edges.values():
            if hasattr(edge, "function"): continue
            from_junction = self.nodes.get(getattr(edge, "from"), None)
            to_junction = self.nodes.get(getattr(edge, "to"), None)
            if from_junction is not None:
                from_junction.outgoing_edges.append(edge)
            if to_junction is not None:
                to_junction.incoming_edges.append(edge)
            setattr(edge, "from", from_junction)
            setattr(edge, "to", to_junction)

        for connection in self.connections:
            if hasattr(connection, "via") and connection.via is not None:
                setattr(connection, "via_lane", self.lanes[connection.via])
            setattr(connection, "from_edge", self.edges.get(getattr(connection, "from"), None))
            if connection.from_edge is not None:
                setattr(connection, "from_lane", connection.from_edge.get_lane(connection.fromLane))
                connection.from_lane.outgoing_connection.append(connection)
            setattr(connection, "to_edge", self.edges.get(getattr(connection, "to"), None))
            if connection.to_edge is not None:
                setattr(connection, "to_lane", connection.to_edge.get_lane(connection.toLane))
                connection.to_lane.incoming_connection.append(connection)

        for node in self.nodes.values():
            if node.type == "internal":
                continue

            for i in node.incLane_ids:
                incLane = self.lanes.get(i, None)
                if incLane is not None:
                    node.incLanes.append(incLane)

            for i in node.intLane_ids:
                intLane = self.lanes.get(i, None)
                if intLane is not None:
                    node.intLanes.append(intLane)

            for edge in self.edges.values():
                edge.link_lane_nodes()

            for connection in self.connections:
                connection.link_lane_nodes()

    def export_lane_nodes(self, path):
        node_id = dict()
        nodes = dict()
        for i, lane_node in enumerate(LaneNodes.EXISTS):
            node_id[lane_node] = i

        for lane_node, ID in node_id.items():
            neighbors = [node_id[neighbor_node] for neighbor_node in lane_node.neighbors]
            nodes[ID] = dict(
                position=lane_node.position.tolist(),
                neighbors=neighbors
            )

        with open(path, 'w') as file:
            json.dump(nodes, file)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    import json
    import time

    now = time.time()
    net = Net(r"C:\Users\Jason\Sumo\2022-09-10-18-45-26\osm.net.xml")
    print(time.time() - now)

    with open("net.pkl", "wb") as file:
        pickle.dump(net, file)

    now = time.time()
    net.link_objects()
    print(time.time() - now)

    # with open(r"F:\Radar Reseach Project\Tracking\SumoNetVis\polygons.json", 'r') as file:
    #     polygons = json.load(file)
    #
    # for layer in polygons:
    #     for obj in polygons[layer]:
    #         obj = np.asarray(obj)
    #         plt.plot(obj[:, 0], obj[:, 1])

    for node in net.nodes.values():
        node.plot()
    plt.show()
