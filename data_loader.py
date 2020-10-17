import torch
import numpy as np
import torch
from renderer import view
import matplotlib.pyplot as plt
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from pytorch3d.ops import sample_points_from_meshes
from utils import get_all_file_paths, process_shape
from pytorch3d.renderer import Textures
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
from file_reader import FileReader
from shape import Shape

from pytorch3d.datasets import (
    collate_batched_meshes
)
sys.path.append(os.path.abspath(''))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    


class ModelNet(Dataset):
    def __init__(self,model_net_path,n_samples=2000):
        self.paths = get_all_file_paths(model_net_path,'off')[0:10]
        self.reader = FileReader()
        data  = [process_shape(Shape(*self.reader.read(path))) for path in tqdm(self.paths)]

        verts = [torch.tensor(x.vertices,dtype=torch.float32) for x in data]
        faces  = [torch.tensor(x.element_dict['triangles'].astype(np.int32),dtype=torch.int32) for x in data]

        self.meshes = Meshes(verts=verts,faces=faces)

        self.samples  = sample_points_from_meshes(self.meshes,n_samples,return_normals=False)
        
        self.classifications = [os.path.basename(os.path.dirname(os.path.dirname(path))) for path in self.paths]
        unique_list = list(set(self.classifications))
        self.class_indexer = { key:unique_list.index(key) for key in unique_list}
        print('Loaded')

    def __len__(self):
        return len(self.classifications)

    def __getitem__(self, index: int):
        return self.samples[index] , self.class_indexer[self.classifications[index]]

    def view(self,index,distance=100.0,elevation = 50.0,azimuth=0.0):
        mesh , classification = self.__getitem__(index)
        print(f'Classified as : {classification}')
        view(mesh,distance,elevation,azimuth)
     

dataset = ModelNet(r'data/ModelNet40')
batch_size = 2
dataset_loader = DataLoader(dataset, batch_size=batch_size)

for epoch in range(10):
    for i, data in enumerate(dataset_loader):
        points = data[0]
        classifications = data[1]
        














