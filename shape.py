import numpy as np
import pyvista
from pathlib import Path
from file_reader import FileReader
import pyvista as pv
import math


class Shape:
    def __init__(self,vertices, element_dict, info):
        self.vertices=vertices.reshape((-1,3))
        self.element_dict = element_dict
        self.info = info
        self.n_triangles = element_dict["triangles"].size/3
        self.n_quads = element_dict["quads"].size
        self.n_vertices = vertices.size/3
        
        
        self.pyvista_mesh = False
    
        
    def view(self):
        self.viewer.process(vertices = self.vertices.flatten() , indices = self.element_dict["triangles"],info=self.info)

    def pyvista_mesh_to_base(self,pyvista_mesh):
        self.element_dict["triangles"] = pyvista_mesh.faces.reshape((-1,4))[:,1:].astype(np.uint32)
        self.vertices = np.array(pyvista_mesh.points.reshape((-1,3))).astype(np.float32)

        self.n_vertices=self.vertices.size/3
        self.n_triangles = self.element_dict["triangles"].size/3
        
        

    def make_pyvista_mesh(self):
        triangles = np.zeros((self.element_dict["triangles"].shape[0],4)) +3
        triangles [:,1:4] = self.element_dict["triangles"]
        triangles = np.array(triangles,dtype=np.int)
        self.pyvista_mesh = pv.PolyData(self.vertices,triangles)


        



  

    
    
    










