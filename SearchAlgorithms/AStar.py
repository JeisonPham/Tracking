import numpy as np
from SearchAlgorithms import SearchObject
import matplotlib.pyplot as plt
import pickle
from SearchAlgorithms.util import color_gradient


class Node:
    ID = 0
    ID_LOOKUP = dict()

    def __init__(self, obj: SearchObject, cost=0, prev_idx=-1, last_move=0, deviation_dist=0, lane_type=None):
        self.obj = obj
        self.g = cost  # g score
        self.prev_idx = prev_idx
        self.last_move = last_move
        self.h = deviation_dist  # h score
        self.lane_type = lane_type

        if obj in Node.ID_LOOKUP:
            self.id = Node.ID_LOOKUP[obj]
        else:
            self.id = Node.ID
            Node.ID_LOOKUP[obj] = self.id
            Node.ID += 1

    def __eq__(self, other):
        return self.obj == other.obj

    def __repr__(self):
        return f"{self.obj.__repr__()}: {self.g}, {self.h}, {self.prev_idx}"

    def calc_heuristic(self, *args, **kwargs):
        return self.obj.calc_heuristic(*args, **kwargs)

    def calc_cost(self, goal):
        return self.obj.calc_cost(goal.obj)


def calc_final_path(goal_reach, start_node, closed_set):
    current = goal_reach
    path = []
    deviation = 0
    while current != start_node:
        path.append(current.obj)
        deviation += current.h
        current = closed_set[current.prev_idx]
    path.append(start_node.obj)
    return path[::-1], deviation


def visualize_closed_set(closed_set):
    viz_map = closed_set[list(closed_set.keys())[0]].obj.OCCUPIED_MAP.copy()

    for node in closed_set.values():
        x, y = node.obj.position
        viz_map[x, y] = 0
    plt.imshow(viz_map, origin='lower')
    plt.show()


def visualize_current_path(current, start, closed_set):
    viz_map = closed_set[0].obj.OCCUPIED_MAP.copy()

    while current != start:
        x, y = current.obj.position
        viz_map[x, y] = 0
        current = closed_set[current.prev_idx]
    x, y = current.obj.position
    viz_map[x, y] = 0
    plt.imshow(viz_map, origin='lower')
    plt.show()


def generate_path(start, goal, *args, **kwargs):
    h_search_threshold = float('inf')
    start_node = Node(start, prev_idx=-1)
    # start_node.obj.plot(plt.gca(), 'black')
    goal_node = Node(goal, prev_idx=-1)
    # goal_node.obj.plot(plt.gca(), 'red')


    farthest_node_index_reached = None

    open_set = dict()
    closed_set = dict()

    open_set[start_node.id] = start_node

    while len(open_set) > 0:
        current_id = None
        min_cost = float('inf')
        for ID, node in open_set.items():
            cost = node.g + node.h + node.calc_cost(goal_node)
            if cost <= min_cost:
                min_cost = cost
                current_id = ID

        current = open_set.pop(current_id)
        # visualize_current_path(current, start_node, closed_set)
        if current == goal_node:
            goal_node = current
            break
        closed_set[current.id] = current
        # visualize_closed_set(open_set)
        if current.h == 0:
            farthest_node_index_reached = current.id

        for node in current.obj.get_next_searches():
            node = Node(node, cost=current.g, prev_idx=current.id)
            node.g += node.calc_cost(current)

            if node.id in closed_set:
                del node
                continue

            node.h = node.calc_heuristic(*args, **kwargs)
            if node.id not in open_set:
                if node.h < h_search_threshold:
                    open_set[node.id] = node
                else:
                    closed_set[node.id] = node
            else:
                if open_set[node.id].g > node.g:
                    del open_set[node.id]
                    open_set[node.id] = node
                elif open_set[node.id].g == node.g:
                    if open_set[node.id].h > node.h:
                        del open_set[node.id]
                        open_set[node.id] = node
                else:
                    del node

    goal_reached = goal_node
    if goal_node.prev_idx == -1 and farthest_node_index_reached is not None:
        nodes = list(closed_set.values())
        nodes = sorted(nodes, key=lambda x: x.calc_cost(goal_node))
        goal_reached = nodes[0]

    path, deviation_dist = calc_final_path(goal_reached, start_node, closed_set)

    Node.ID_LOOKUP.clear()
    Node.ID = 0
    return path, deviation_dist
