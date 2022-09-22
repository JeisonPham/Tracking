import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import matplotlib

RADIUS = 10


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


class Lane:
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.parent_edge = None
        self.index = int(self.index)
        self.incoming_connection = []
        self.outgoing_connection = []
        self.end = [float(x) for x in self.shape.split(" ")[-1].split(",")]
        self.start = [float(x) for x in self.shape.split(" ")[0].split(",")]

    def __repr__(self):
        return self.id


class Connection:
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.fromLane = int(self.fromLane)
        self.toLane = int(self.toLane)


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
