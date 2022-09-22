from MissionPlanner.NetNodes import Net
import numpy as np
import matplotlib.pyplot as plt
import json
from shapely.geometry import Polygon, Point


class MissionManager(object):
    def __init__(self, net_file, polygon_file, navigation_method="bfs"):
        self.net = Net(net_file)
        self.net.link_objects()

        with open(polygon_file, 'r') as file:
            polygons = json.load(file)
            for key, value in polygons['junction'].items():
                polygon = Polygon(value)
                setattr(self.net.nodes[key], "polygon", polygon)

            for key, value in polygons['lane'].items():
                lane_polygon = Polygon(value)
                setattr(self.net.lanes[key], "polygon", lane_polygon)

                temp = []
                for marking in polygons['lane_markings'][key]:
                    lane_marking = Polygon(marking)
                    temp.append(lane_marking)
                setattr(self.net.lanes[key], "marking_polygon", temp)

        self._navigation_method = navigation_method
        self.current_lane = None

    def set_node(self, position, node_type="start"):
        point = Point(position)
        found = False
        for key, lane in self.net.lanes.items():
            if hasattr(lane, "polygon") and lane.polygon.contains(point):
                found = True
                break

        if not found:
            for key, junction in self.net.nodes.items():
                if hasattr(junction, "polygon") and junction.polygon.contains(point):
                    lane = junction.incLanes[0]

        if node_type == "start":
            self.current_lane = lane
        else:
            setattr(self, "goal_lane", lane)

    def generate_path(self):
        if self._navigation_method == "bfs":
            setattr(self, "path", self._bfs())

    def target(self, position):
        point = Point(position)
        if self.current_lane.polygon.contains(point):
            return self.current_lane.end
        else:
            return self.current_lane.start

    def target_list(self):
        targets = []
        for lane in self.path:
            targets.append(lane.polygon.centroid.coords[0])

        return targets

    def update(self, position):
        point = Point(position)

        if not self.current_lane.contains(point):
            self.current_lane = self.path.pop(0)

    def _bfs(self):
        queue = [[self.current_lane]]
        visited = set()
        while len(queue) > 0:
            size = len(queue)
            for _ in range(size):
                lanes = queue.pop(0)
                if lanes[-1] in visited:
                    continue
                elif lanes[-1] == self.goal_lane:
                    return lanes

                visited.add(lanes[-1])
                current_lane = lanes[-1]
                for connection in current_lane.outgoing_connection:
                    temp = lanes.copy()
                    temp.append(connection.to_lane)
                    queue.append(temp)

        raise Exception


if __name__ == "__main__":
    manager = MissionManager(r"C:\Users\Jason\Sumo\2022-09-10-18-45-26\osm.net.xml",
                             r"F:\Radar Reseach Project\Tracking\SumoNetVis\polygons.json")
    manager.set_node([308.93, 580.42])
    manager.set_node([937.51, 923.72], "goal")
    manager.generate_path()
    manager.target()
