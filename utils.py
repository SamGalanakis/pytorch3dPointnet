import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path

def get_all_file_paths(directory,extension):
    '''Get all file paths of given extension for files under given directory '''
    file_paths = []
    for root, dirs, files in os.walk(Path(directory)):
        for file in files:
            if file.endswith(extension):
                
                file_paths.append(os.path.join(root, file))
    return file_paths

def process_shape(shape):
    shape.make_pyvista_mesh()
   # while shape.pyvista_mesh.n_points<2000:
        #shape.pyvista_mesh.subdivide(2,inplace=True)
    shape.pyvista_mesh_to_base(shape.pyvista_mesh)
    return shape


def view_pointcloud(points):
    
    points = points.cpu().numpy()
  
    ax = plt.axes(projection='3d')
    xyz=points
    color = np.array([1,1,1])
  
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c = (color/255).reshape(1,-1), s=0.1)
    plt.show()



