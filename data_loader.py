import torch
import numpy as np
import torch
from renderer import view
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from pytorch3d.ops import sample_points_from_meshes
from utils import get_all_file_paths, process_shape, view_pointcloud
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
from utils import farthest_point_sample
import torch
import pickle


sys.path.append(os.path.abspath(''))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    


class ModelNet(Dataset):
    def __init__(self,model_net_path,n_samples=2000,lazy=True,mode='train'):
        assert mode in ['train','test'], 'Invalid mode'
        print(f'Dataset initialized in {mode} mode!')
        self.mode = mode
        self.lazy = lazy
        self.n_samples= n_samples
        self.paths = get_all_file_paths(model_net_path,'off')
        self.paths_test = sorted([x for x in self.paths if 'test' in x])
        self.paths_train = sorted(list(set(self.paths).difference(set(self.paths_test))))

        if self.mode == 'train':
            self.paths = self.paths_train
        elif self.mode == 'test':
            self.paths = self.paths_test
        
        

        classifications = [os.path.basename(os.path.dirname(os.path.dirname(path))) for path in self.paths]
        

        
        unique_list = sorted(list(set(classifications) ))
        
        self.class_indexer = { key:unique_list.index(key) for key in unique_list}

        if not lazy:
            print(f"Reading .off files")
            save_path = f"data/data_{mode}.p"
            save_exists = os.path.isfile(save_path)
            if not save_exists:
                data = [parse_off(path) for path in tqdm(self.paths)]
                print("Saving data to file")
                pickle.dump( data, open( save_path, "wb" ) )
            else:
                print("Reading from save")
                data = pickle.load( open( save_path, "rb" ) )

            

            verts = [torch.tensor(x[0],dtype=torch.float32) for x in data]

            save_path = f"data/samples_{n_samples}_{mode}.p"
            save_exists = os.path.isfile(save_path)
            if not save_exists:
                self.samples = [farthest_point_sample(vertices,n_samples) for vertices in tqdm(verts)]
                print("Saving samples to file")
                pickle.dump( self.samples, open( save_path, "wb" ) )
            else:
                print("Reading samples from save")
                self.samples = pickle.load( open( save_path, "rb" ) )
            


            
            
            #faces  = [torch.tensor(x[1],dtype=torch.int32) for x in data]
            
            
            self.classifications=classifications

            #self.samples = []
            #self.classifications =[]
            #failed_samplings = 0
            #Also add the classifications to make sure they match after removing problematic models
            # print("Sampling models...")
            # for vert_data,face_data,classification in tqdm(zip(verts,faces,classifications)):
            #     try:
            #         vertices_sample = sample_points_from_meshes(Meshes(verts=[vert_data],faces=[face_data]),return_normals=False,num_samples=self.n_samples)
            #         self.samples.append(vertices_sample.squeeze())
            #         self.classifications.append(classification)
            #     except:
            #         failed_samplings +=1
            #         print(f'Failed to load, total: {failed_samplings}')
            #         continue
            # print(f"Failed to sample {failed_samplings} of {len(verts)}" )
            

     
    

    def __len__(self):
            return len(self.samples)
     
    def get_item_lazy(self,index):
       
        data = parse_off(self.paths[index])
        mesh = Meshes([torch.tensor(data[0],dtype=torch.float32)],[torch.tensor(data[1],dtype=torch.int)])
        try:
            return torch.squeeze(sample_points_from_meshes(mesh,return_normals=False,num_samples=self.n_samples)),self.class_indexer[self.classifications[index]]
        except:
            print(f"Could not sample index {index}")
            #Make recursive call with random model if one of the models can't be sampled
            return self.__getitem__(random.randint(0,self.__len__()))

    def __getitem__(self, index: int):
     
        if self.lazy:
            return self.get_item_lazy(index)
        return self.samples[index], self.class_indexer[self.classifications[index]]

    def view(self,index):
        
        points , classification_val = self.__getitem__(index)
        print(f'Classified as : {self.classifications[index]}')
        view_pointcloud(points)
     




        














