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
from off_parser import parse_off
from shape import Shape
import random

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
    def __init__(self,model_net_path,n_samples=2000,lazy=True):
        self.lazy = lazy
        self.n_samples= n_samples
        self.paths = get_all_file_paths(model_net_path,'off')
        
        self.paths_test = [x for x in self.paths if 'test' in x]
        self.paths_train = set(self.paths).difference_update(set(self.paths_test))

        self.classifications_test = [os.path.basename(os.path.dirname(os.path.dirname(path))) for path in self.paths_test]
        self.classifications_train= [os.path.basename(os.path.dirname(os.path.dirname(path))) for path in self.paths_train]

        combined_classifications = set(self.classifications_test).update(set(self.classifications_train))
        unique_list = sorted(list(combined_classifications ))
        
        self.class_indexer = { key:unique_list.index(key) for key in unique_list}

        if not lazy:
            data_train  = [parse_off(path) for path in tqdm(self.paths_train)]
            data_test = [parse_off(path) for path in tqdm(self.paths_test)]

            verts_train = [torch.tensor(x[0],dtype=torch.float32) for x in data_train]
            verts_test = [torch.tensor(x[0],dtype=torch.float32) for x in data_test]
            
            faces_train  = [torch.tensor(x[1],dtype=torch.int32) for x in data_train]
            faces_test  = [torch.tensor(x[1],dtype=torch.int32) for x in data_test]

            
            self.samples_train = sample_points_from_meshes(Meshes(verts=verts_train,faces=faces_train),return_normals=False,num_samples=self.n_samples)
            self.samples_test = sample_points_from_meshes(Meshes(verts=verts_test,faces=faces_test),return_normals=False,num_samples=self.n_samples)

        #Set mode to train by default

        self.set_mode('train')
       
    def set_mode(self,mode):
        print(f"Setting mode as {mode}")

        self.mode = mode
        assert mode in ['train','test'], 'Invalid mode not  train or test!'

        if mode == 'train':
            
            self.curr_classifications = self.classifications_train
            self.curr_paths = self.paths_train
            if  self.lazy:
                self.curr_samples = self.samples_train
        else:
            self.curr_classifications = self.classifications_test
            self.curr_paths = self.paths_test
            if  self.lazy:
                self.curr_samples = self.samples_test




    def __len__(self):
        if self.train:
            return len(self.classifications_train)
        else:
            return len(self.classifications_test)
    def get_item_lazy(self,index):
       
        data = parse_off(self.curr_paths[index])
        mesh = Meshes([torch.tensor(data[0],dtype=torch.float32)],[torch.tensor(data[1],dtype=torch.int)])
        try:
            return torch.squeeze(sample_points_from_meshes(mesh,return_normals=False,num_samples=self.n_samples)),self.class_indexer[self.curr_classifications[index]]
        except:
            print(f"Could not sample index {index}")
            #Make recursive call with random model if one of the models can't be sampled
            return self.__getitem__(random.randint(0,self.__len__()))

    def __getitem__(self, index: int):
     
        if self.lazy:
            return self.get_item_lazy(index)
        return self.curr_samples[index], self.class_indexer[self.curr_classifications[index]]

    def view(self,index,distance=100.0,elevation = 50.0,azimuth=0.0):
        mesh , classification = self.__getitem__(index)
        print(f'Classified as : {classification}')
        view(mesh,distance,elevation,azimuth)
     

# dataset = ModelNet(r'data/ModelNet40')
# batch_size = 2
# dataset_loader = DataLoader(dataset, batch_size=batch_size)


        














