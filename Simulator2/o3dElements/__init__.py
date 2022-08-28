import open3d
import sys

cuda = "cuda" in sys.argv
import open3d.cuda.pybind.visualization.rendering as rendering
import open3d.cuda.pybind.visualization.gui as gui
import open3d.cuda.pybind.t.geometry as geometry

if cuda and open3d.__DEVICE_API__ == 'cuda':

    device = open3d.core.Device("CUDA:0")
    print("Using Cuda")
else:
    device = open3d.core.Device("CPU:0")