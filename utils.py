import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path
import torch 
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
   
    N, D = point.shape
    xyz = point[:,:3]
    
    centroids = torch.zeros((npoint,))
    
    distance = (torch.ones((N,)) * 1e10)
   
    farthest = torch.randint(0, N, (1,)).item()
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance


        #dist_select = torch.masked_select(dist,mask) 
       # torch.masked_select(distance,mask) = dist_select 
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance,-1)
   
    point = point[centroids.long()]
    return point



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



