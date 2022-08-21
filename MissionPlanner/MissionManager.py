from .MissionNode import MissionNode
import numpy as np
import matplotlib.pyplot as plt


class MissionManager(object):
    def __init__(self, starting_position, radius, node_file):
        self.nodes = MissionNode.create_nodes_from_file(node_file, radius)
        self.current_node = self.nodes[[key for key, value in sorted(self.nodes.items(), key=lambda x: np.linalg.norm(starting_position[:-1].flatten() -
                                                                                       x[1].starting_position))][0]]
        self._goal = None
        self._visited = []

    def set_goal(self, goal):
        self._goal = self.nodes[[key for key, value in sorted(self.nodes.items(), key=lambda x: np.linalg.norm(goal -
                                                                                   x[1].starting_position))][0]]

        self._path = MissionNode.A_star(self.current_node, self._goal)

    def update_node(self, vehicle):
        if np.linalg.norm(vehicle - self.current_node.starting_position) < self.current_node.radius:
            self._visited.append(self.current_node)
            self.current_node = self._path.pop(0)

            if self.current_node == self._goal:
                print("Reached Goal")

            return True
        return False

    def plot(self, transform_func, player):
        if len(self._visited) > 0:
            visited = np.array([[*node.starting_position, 0, 0] for node in self._visited]).reshape(-1, 4)
            visited = transform_func(visited, player)
            plt.scatter(visited[:, 1], visited[:, 0], c='g', marker='x')

        if len(self._path) > 0:
            path = np.array([[*node.starting_position, 0, 0] for node in self._path]).reshape(-1, 4)
            path = transform_func(path, player)
            plt.scatter(path[:, 1], path[:, 0], c='black', marker='o')


if __name__ == "__main__":
    manager = MissionManager(np.array([858, 514]), 10, "nodes.json")
