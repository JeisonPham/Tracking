import open3d as o3d
try:
    from Models import Base
except:
    import Simulator2.Models.Base as Base

import numpy as np
import open3d.visualization.rendering as rendering


class MapModel(Base):
    def __init__(self, file: str, l: int = 5, scene: rendering.Open3DScene = None):
        if file.endswith(".ply"):
            pc = o3d.io.read_point_cloud(file)
            points = np.asarray(pc.points)

            points = points[points[:, 0] < 390]

            min_x = min(points[:, 0])
            max_y = max(points[:, 2])
            offset = np.array([-min_x, 0, -max_y])
            points = points + offset
            points[:, 2] = -points[:, 2]
            points = points[points[:, 1] <= 1e-4]
            pc.points = o3d.utility.Vector3dVector(points)
            mesh = pc

            # # # o3d.visualization.draw_geometries([pc])
            #
            # v_size = round(max(pc.get_max_bound() - pc.get_min_bound()) * 0.005, 4)
            # print(v_size)
            # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=4)
            # voxels = voxel_grid.get_voxels()
            #
            # index = np.array([v.grid_index for v in voxels])
            # min_y = min(index[:, 1])
            # max_y = max(index[:, 1])
            #
            # gradient = Color.gradient_helper(colors=[Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE, Color.PURPLE])
            #
            # layers = {x + 1: o3d.geometry.TriangleMesh() for x in range(l)}
            #
            # for v in voxels:
            #     cube = o3d.geometry.TriangleMesh.create_box()
            #     layer = (v.grid_index[1] - min_y) / (max_y - min_y)
            #     color = gradient((v.grid_index[1] - min_y) / (max_y - min_y))
            #     cube.paint_uniform_color(color)
            #     cube.translate(v.grid_index, relative=False)
            #     for key in layers.keys():
            #         if layer * l <= key:
            #             layers[key] += cube
            #             break
            #
            # mesh = o3d.geometry.TriangleMesh()
            # for key, value in layers.items():
            #     value.translate([0.5, 0.5, 0.5], relative=True)
            #     value.scale(4, [0, 0, 0])
            #     value.translate(voxel_grid.origin, relative=True)
            #     value.merge_close_vertices(0.0000001)
            #
            #     o3d.io.write_triangle_mesh(f"MapFiles/Layer_{key}.obj", value)
            #     mesh += value
        else:
            mesh = o3d.geometry.TriangleMesh()
            for layer in range(l):
                voxel = o3d.io.read_triangle_mesh(f"{file}/Layer_{layer + 1}.obj")
                mesh += voxel

        # o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

        super().__init__(name="Map",
                         geometry=mesh,
                         material=rendering.MaterialRecord(),
                         scene=scene)

        self.position = np.around(self.geometry.get_center(), 3)

    @staticmethod
    def read_pc(file_path):
        pc = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pc.points)

        points = points[points[:, 0] < 390]

        min_x = min(points[:, 0])
        max_y = max(points[:, 2])
        offset = np.array([-min_x, 0, -max_y])
        points = points + offset
        points[:, 2] = -points[:, 2]
        points = points[points[:, 1] <= 1e-4]

        return points[:, (0, 2)]

    def step(self, frame):
        return True




if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # map = MapModel("F:\Radar Reseach Project\Tracking\Data\downtown_SD_10_7.ply", l=10, scene=None)
    # map = MapModel("MapFiles", l=1, scene=None)
