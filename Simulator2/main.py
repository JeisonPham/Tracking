from BaseSimulator import BaseSimulator
import Simulator2.Nodes as Nodes
import open3d as o3d
import numpy as np
import Simulator2.o3dElements

if __name__ == "__main__":
    sim = BaseSimulator("Testing", 1920, 1080)
    comp = Nodes.CompNode("F:\Radar Reseach Project\Tracking\Data\downtown_SD_10thru_50count_with_cad_id.csv", "F:\Radar Reseach Project\Tracking\Data\downtown_SD_10_7.ply")
    sim.register_node(comp)
    sim.register_node(Nodes.MapNode("F:\Radar Reseach Project\Tracking\Simulator2\Models\MapFiles", layers=1))
    sim.register_node(Nodes.PlaybackNode(comp, True))
    sim.register_node(Nodes.CreateNode(True))
    sim.register_node(Nodes.MissionNode(np.asarray([0, 0, 0]), 15, "../MissionPlanner/nodes.json"))
    sim.register_node(Nodes.EgoNode([0, 0, 0], 0))
    sim.register_node(Nodes.ImageNode(True))
    sim.run()