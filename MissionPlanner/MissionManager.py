import matplotlib.patches

from MissionPlanner.NetNodes import Net, LaneNodes
from PathManager import PathManager
import numpy as np
import matplotlib.pyplot as plt
import json
from shapely.geometry import Polygon, Point
from SearchAlgorithms import SearchObject, AStar
from GridPlanning import LocalPlannerNode
import cv2


def get_grid(point_cloud_range, voxel_size):
    lower = np.array(point_cloud_range[:(len(point_cloud_range) // 2)])
    upper = np.array(point_cloud_range[(len(point_cloud_range) // 2):])

    dx = np.array(voxel_size)
    bx = lower + dx / 2.0
    nx = ((upper - lower) / dx).astype(int)

    return dx, bx, nx


def get_corners(box, lw):
    l, w = lw
    simple_box = np.array([
        [-l / 2., -w / 2.],
        [l / 2., -w / 2.],
        [l / 2., w / 2.],
        [-l / 2., w / 2.],
    ])
    return simple_box + np.array(box)


class GridTransform:
    def __init__(self, center):
        self.dx, self.bx, (self.nx, self.ny) = get_grid([-5, -10,
                                                         15, 10],
                                                        [20 / 256, 20 / 256])
        self.center = center

    def __call__(self, position):
        pts = position - self.center
        pts = np.round(
            (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
        ).astype(np.int32)
        return pts

    def reverse(self, position):
        position = position * self.dx[:2] + self.bx[:2] - self.dx[:2] / 2.
        return position + self.center


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

    def get_node(self, position):
        dist = np.linalg.norm([x.position - position for x in LaneNodes.EXISTS], axis=1)
        index = np.argsort(dist)[0]
        return LaneNodes.EXISTS[index]

    def generate_path(self, navigation_method="bfs"):
        if navigation_method == "bfs":
            setattr(self, "path", self._bfs())
        elif navigation_method == "AStar":
            path = AStar.generate_path(MissionNodeWrapper(self.start_node),
                                       MissionNodeWrapper(self.goal_node))[0]

            path = [x.lane_node for x in path]
            setattr(self, "path", path)

    def create_occupancy_grid(self, position, occupied_information):
        position = np.array(position)
        nodes = []
        for junction in self.net.nodes.values():
            if hasattr(junction, 'polygon'):
                if any(np.linalg.norm(np.array([*junction.polygon.exterior.xy]).T - position, axis=1) <= 60):
                    nodes.append(junction.polygon)
        for lane in self.net.lanes.values():
            if hasattr(lane, 'polygon'):
                if any(np.linalg.norm(np.array([*lane.polygon.exterior.xy]).T - position, axis=1) <= 60):
                    nodes.append(lane.polygon)

        gridTransform = GridTransform(position)
        occupancy_map = np.zeros((gridTransform.nx, gridTransform.ny))

        for node in nodes:
            temp_map = np.zeros((gridTransform.nx, gridTransform.ny))
            pts = np.array([*node.exterior.xy]).T
            pts = gridTransform(pts)
            cv2.fillPoly(temp_map, [pts[:, (1, 0)]], color=1.0)
            cv2.polylines(temp_map, [pts[:, (1, 0)]], color=2.0, thickness=2, isClosed=True)

            for info in occupied_information:
                p = Point(info['position'])
                if node.contains(p):
                    pts = gridTransform(info['position'])
                    temp_map = cv2.circle(temp_map, pts[::-1], radius=51, color=0.0, thickness=-1)

            occupancy_map = np.clip(occupancy_map + temp_map, 0., 1.)

        plt.imshow(occupancy_map, origin='lower')
        plt.show()

        return occupancy_map, gridTransform

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


if __name__ == "__main__":
    manager = MissionManager(r"E:\2022-09-12-16-57-35\osm.net.xml",
                             r"E:\Radar Reseach Project\Tracking\SumoNetVis\polygons.json")

    manager.set_node([308.93, 580.42])
    manager.set_node([480, 580], "goal")
    manager.plot([308.93, 580.42], hide_nodes=True)
    plt.scatter(308.93, 580.42)
    plt.show()

    manager.generate_path("AStar")

    path = [x for x in manager.path]

    pathManager = PathManager(path, AStar.generate_path, MissionNodeWrapper)

    local_path = []
    next_motion = None

    occupied_info = [{
        'position': [320.93, 583]
    }, {
        'position': [340.93, 582]
    }
    ]

    it = pathManager.get_path()
    index = -1
    for waypoint_start, waypoint_goal in it:
        index += 1
        # fig = plt.figure(figsize=(20, 20))
        omap, transform = manager.create_occupancy_grid(waypoint_start.position, occupied_info)

        if next_motion is None:
            next_motion = LocalPlannerNode.motion_calculator(waypoint_start.position, waypoint_goal.position)

        start = LocalPlannerNode(*transform(waypoint_start.position), list(waypoint_start.neighbors), transform, omap,
                                 next_motion)
        goal = LocalPlannerNode(*transform(waypoint_goal.position))
        manager.plot((waypoint_start.position + waypoint_goal.position) / 2., hide_nodes=True)
        for info in occupied_info:
            x = info['position'][0]
            y = info['position'][1]

            patch = matplotlib.patches.Rectangle([x - 1.5, y - 0.5], 3, 1, color='black')
            plt.gca().add_patch(patch)

        local_path += AStar.generate_path(start, goal)[0]
        for local in local_path:
            local.plot(plt.gca(), 'blue')

        path = pathManager.numpy_path()
        plt.plot(path[:, 0], path[:, 1], 'r-->')
        plt.savefig(f"Output/{index:05d}.png")
        plt.clf()

        dist = float('inf')
        final_position = transform.reverse(local_path[-1].position)
        for neighbor in waypoint_start.neighbors:
            if np.linalg.norm(neighbor.position - final_position) < dist:
                dist = np.linalg.norm(neighbor.position - final_position)
                final_waypoint = neighbor

        final_waypoint.position = transform.reverse(local_path[-1].position)
        if final_waypoint != waypoint_goal:
            pathManager.regenerate_path(final_waypoint)
        pathManager.start = final_waypoint

        next_motion = local_path[-1].current_move

    # TODO: Test scenario where vehicle must wait before proceeding
    # TODO: Test left turn with obstruction
    # TODO: Test Right turn with obstruction
    # TODO: Integrate with simulator or use another one

