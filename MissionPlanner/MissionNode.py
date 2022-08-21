import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq


class MissionNode(object):
    def __init__(self, starting_position, radius, name=None):
        self.name = name
        self.starting_position = np.array(starting_position).flatten()
        self.radius = radius
        self.nodelist = {}

        self.h = 0
        self.g = 0

    @property
    def f(self):
        return self.h + self.g

    def add_to_nodelist(self, node, weight):
        self.nodelist[node] = weight

    def plot(self):
        style = "Simple, tail_width=0.5, head_width=4, head_length=8"

        line = plt.scatter(self.starting_position[0], self.starting_position[1], s=self.radius)
        kw = dict(arrowstyle=style, color="k")
        for node in self.nodelist:
            a = patches.FancyArrowPatch(self.starting_position, node.starting_position,
                                        connectionstyle='arc3,rad=0.1', **kw)
            plt.gca().add_patch(a)

    def __lt__(self, other):
        return self.f < other.f

    def __repr__(self):
        return self.name

    @staticmethod
    def create_nodes_from_file(txtfile, radius, metric='distance'):
        with open(txtfile, 'r') as f:
            info = json.load(f)

        node_names = {}
        for key in info.keys():
            node_names[key] = MissionNode([info[key]['x'], info[key]['y']], radius, name=key)

        for key in info.keys():
            for nn in info[key]["nodes"]:
                if metric == 'distance':
                    weight = np.linalg.norm(node_names[nn].starting_position - node_names[key].starting_position)
                node_names[key].add_to_nodelist(node_names[nn], weight)

        # for value in node_names.values():
        #     value.plot()
        #
        # plt.show()

        return node_names

    @staticmethod
    def A_star(start, stop):
        start.h = np.linalg.norm(start.starting_position - stop.starting_position)
        open_list = set([start])
        closed_list = set()

        parents = {start: start}

        while len(open_list) > 0:
            node = sorted(open_list)[0]
            open_list.remove(node)

            if node == stop:
                path = []

                while parents[node] != node:
                    path.append(node)
                    node = parents[node]
                path.append(start)
                path.reverse()
                return path

            closed_list.add(node)

            for m, weight in node.nodelist.items():
                m.g = node.g + weight
                m.h = np.linalg.norm(m.starting_position - stop.starting_position)

                if m not in closed_list and m not in open_list:
                    parents[m] = node
                    open_list.add(m)
                elif m in open_list:
                    if m.f < node.f:
                        parents[m] = node




import csv

if __name__ == "__main__":
    MissionNodes = MissionNode.create_nodes_from_file('nodes.json', radius=50)
    plt.show()

    path = MissionNode.A_star(MissionNodes['Node11'], MissionNodes['Node32'])
    print(path)
    # nodes = {}
    # with open("temp.csv", 'r') as file:
    #     csv_reader = csv.reader(file)
    #     for row in csv_reader:
    #         nodes["Node" + row[0]] = {
    #             'x': float(row[1]),
    #             'y': float(row[2]),
    #             'nodes': []
    #         }
    # for key, value in nodes.items():
    #     pos = np.array([value['x'], value['y']])
    #     others = {k: np.array([v['x'], v['y']]) for k, v in nodes.items() if k != key}
    #     others = {k: v for k, v in sorted(others.items(), key = lambda x: np.linalg.norm(x[1] - pos))}
    #
    #     for k, v in others.items():
    #         if len(nodes[key]['nodes']) < 4:
    #             diff = v - pos
    #             current_angle = abs(np.rad2deg(np.arctan2(diff[1], diff[0])) % 90)
    #             if 10 < current_angle < 80:
    #                 break
    #
    #             nodes[key]['nodes'].append(k)
    #         else:
    #             break
    #
    # with open('nodes.json', 'w') as file:
    #     json.dump(nodes, file)
