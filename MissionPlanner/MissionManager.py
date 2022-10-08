from MissionPlanner.NetNodes import Net, LaneNodes
import numpy as np
import matplotlib.pyplot as plt
import json
from shapely.geometry import Polygon, Point
from SearchAlgorithms import SearchObject, AStar
from GridPlanning import LocalPlannerNode


class MissionNodeWrapper(SearchObject):
    def __init__(self, lane_node):
        self.lane_node = lane_node

    def __hash__(self):
        return self.lane_node.__hash__()

    def calc_heuristic(self, *args, **kwargs):
        return 0

    def calc_cost(self, goal):
        return np.linalg.norm(self.lane_node.position - goal.lane_node.position)

    def __eq__(self, other):
        return self.lane_node == other.lane_node

    def get_next_searches(self):
        for neighbor in self.lane_node.neighbors:
            yield MissionNodeWrapper(neighbor)


class MissionManager(object):
    def __init__(self, net_file, polygon_file):
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

        self.current_lane = None

        # self.net.export_lane_nodes("../LocalPlanner/waypoints.json")

    def set_node(self, position, node_type="start"):
        dist = np.linalg.norm([x.position - position for x in LaneNodes.EXISTS], axis=1)
        index = np.argsort(dist)[0]

        setattr(self, f"{node_type}_node", LaneNodes.EXISTS[index])

    def generate_path(self, navigation_method="bfs"):
        if navigation_method == "bfs":
            setattr(self, "path", self._bfs())
        elif navigation_method == "AStar":
            path = AStar.generate_path(MissionNodeWrapper(self.start_node),
                                       MissionNodeWrapper(self.goal_node))[0]

            path = [x.lane_node for x in path]
            setattr(self, "path", path)

    # def target(self, position):
    #     point = Point(position)
    #     if self.current_lane.polygon.contains(point):
    #         return self.current_lane.end
    #     else:
    #         return self.current_lane.start

    def plot(self, position, hide_nodes=False):
        for junction in self.net.nodes.values():
            junction.draw(position)
        for edge in self.net.edges.values():
            edge.draw(position)
        for lane in self.net.lanes.values():
            lane.draw(position, hide_nodes)
        for connection in self.net.connections:
            connection.draw(position, hide_nodes)

        plt.xlim([position[0] - 60, position[0] + 60])
        plt.ylim([position[1] - 60, position[1] + 60])

    # def target_list(self):
    #     targets = []
    #     for lane in self.path:
    #         targets.append(lane.polygon.centroid.coords[0])
    #
    #     return targets
    #
    # def update(self, position):
    #     point = Point(position)
    #
    #     if not self.current_lane.contains(point):
    #         self.current_lane = self.path.pop(0)
    #
    # def _bfs(self):
    #     queue = [[self.current_lane]]
    #     visited = set()
    #     while len(queue) > 0:
    #         size = len(queue)
    #         for _ in range(size):
    #             lanes = queue.pop(0)
    #             if lanes[-1] in visited:
    #                 continue
    #             elif lanes[-1] == self.goal_lane:
    #                 return lanes
    #
    #             visited.add(lanes[-1])
    #             current_lane = lanes[-1]
    #             for connection in current_lane.outgoing_connection:
    #                 temp = lanes.copy()
    #                 temp.append(connection.to_lane)
    #                 queue.append(temp)
    #
    #     raise Exception


if __name__ == "__main__":
    manager = MissionManager(r"F:\2022-09-12-16-57-35\osm.net.xml",
                             r"F:\Radar Reseach Project\Tracking\SumoNetVis\polygons.json")
    fig = plt.figure(figsize=(20, 20))
    manager.plot([308.93, 580.42], hide_nodes=True)
    manager.set_node([308.93, 580.42])
    manager.set_node([937.51, 923.72], "goal")
    manager.generate_path("AStar")

    path = np.array([x.position for x in manager.path])
    start = LocalPlannerNode(311.05000, 317.61600)

    goal = LocalPlannerNode(317.61600, 324.18200)

    local_path = AStar.generate_path(start, goal)[0]
    for local in local_path:
        local.plot(plt.gca())
    plt.plot(path[:, 0], path[:, 1], 'r-->')
    plt.show()

    # plt.show()
