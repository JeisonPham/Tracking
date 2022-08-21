import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def get_map(file):
    pc = o3d.io.read_point_cloud(file)
    points = np.asarray(pc.points)

    points = points[points[:, 0] < 390]

    min_x = min(points[:, 0])
    max_y = max(points[:, 2])
    offset = np.array([-min_x, 0, -max_y])
    points = points + offset
    points[:, 2] = -points[:, 2]
    points = points[points[:, 1] <= 1e-4]
    return points


if __name__ == "__main__":

    from scipy.spatial import Delaunay
    import numpy as np


    def alpha_shape(points, alpha, only_outer=True):
        """
        Compute the alpha shape (concave hull) of a set of points.
        :param points: np.array of shape (n,2) points.
        :param alpha: alpha value.
        :param only_outer: boolean value to specify if we keep only the outer border
        or also inner edges.
        :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
        the indices in the points array.
        """
        assert points.shape[0] > 3, "Need at least four points"

        def add_edge(edges, i, j):
            """
            Add a line between the i-th and j-th points,
            if not in the list already
            """
            if (i, j) in edges or (j, i) in edges:
                # already added
                assert (j, i) in edges, "Can't go twice over same directed edge right?"
                if only_outer:
                    # if both neighboring triangles are in shape, it is not a boundary edge
                    edges.remove((j, i))
                return
            edges.add((i, j))

        tri = Delaunay(points)
        edges = set()
        # Loop over triangles:
        # ia, ib, ic = indices of corner points of the triangle
        for ia, ib, ic in tri.simplices:
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]
            # Computing radius of triangle circumcircle
            # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
            a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
            c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
            s = (a + b + c) / 2.0
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_r = a * b * c / (4.0 * area)
            if circum_r < alpha:
                add_edge(edges, ia, ib)
                add_edge(edges, ib, ic)
                add_edge(edges, ic, ia)
        return edges

    points = get_map("F:\Radar Reseach Project\Tracking\Data\downtown_SD_10_7.ply")
    alpha_shape = alphashape.alphashape(points)

    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1])
    ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
    plt.show()
