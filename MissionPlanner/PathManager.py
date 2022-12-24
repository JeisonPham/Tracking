import numpy as np


class PathManager:
    def __init__(self, starting_path, path_finding, wrapper):
        self.path = starting_path
        self.path_finding = path_finding
        self.wrapper = wrapper

    def regenerate_path(self, new_start):
        path = self.path_finding(self.wrapper(new_start), self.wrapper(self.path[len(self.path) // 2]))[0]
        path = [x.lane_node for x in path]
        self.path = path[1:] + self.path[len(self.path) // 2 + 1:]

    def get_path(self):
        self.start = self.path.pop(0)

        while len(self.path) > 0:
            # need to reference self.path in some manner. Weird error where path resets if not referenced
            # probably a weird error with generator functions
            print(len(self.path))
            goal = self.path.pop(0)
            yield self.start, goal

    def numpy_path(self):
        return np.array([x.position for x in self.path]).reshape(-1, 2)
