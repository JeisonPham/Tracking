import numpy as np
import json


class _Waypoint:
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.neighbor_nodes = []
        self.position = np.array(self.position)
        self.occupied = False

    def __repr__(self):
        return self.id


def create_waypoint_queue(path):
    with open(path, 'r') as file:
        data = json.load(file)

    waypoint_ids = {}
    for id, info in data.items():
        info['id'] = int(id)
        waypoint_ids[int(id)] = _Waypoint(**info)

    for waypoint in waypoint_ids.values():
        waypoint.neighbor_nodes = [waypoint_ids[ID] for ID in waypoint.neighbors]

    return list(waypoint_ids.values())


if __name__ == "__main__":
    create_waypoint_queue('waypoints.json')
