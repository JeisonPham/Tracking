import open3d as o3d
from Simulator2.o3dElements import gui, rendering, geometry, device
import Simulator2.Nodes as Node
import numpy as np


class MapNode(Node.ModelNode):
    def __init__(self, map_file: str, layers: int, *args, **kwargs):
        mesh = o3d.geometry.TriangleMesh()
        for layer in range(layers):
            voxel = o3d.io.read_triangle_mesh(f"{map_file}/Layer_{layer + 1}.obj")
            mesh += voxel
        mesh.compute_vertex_normals()
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultLit'

        mesh = geometry.TriangleMesh.from_legacy(mesh, device=device)

        super().__init__(name="Map",
                         geometry=mesh,
                         material=mat)

        center = self.geometry.get_center()
        if center.is_cuda:
            center = center.cpu()
        self.position = np.around(center.numpy(), 3)

    def on_start(self):
        super().on_start()

        # self.scene.add_geometry(self.name, self.geometry, self.material)
        self._sim.scene.look_at(self.position, self.position + np.array([0, 40, 0]), [0, 1, 0])

