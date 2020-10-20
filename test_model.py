
from data_loader import ModelNet
from torch.utils.data import DataLoader
from model import Classifier
from tqdm import tqdm 
import torch
import os 
import numpy as np

model_path = 'wandb/run-20201020_160716-239oq10x/files/260model.pt'
dataset_test = ModelNet(r'data/ModelNet40',lazy=False,mode='test',n_samples=2000)
dataset_train = ModelNet(r'data/ModelNet40',lazy=False,mode='train',n_samples=2000)
dataset_loader_test = DataLoader(dataset_test, batch_size=int(len(dataset_test)/20),shuffle=True,num_workers=4)
dataset_loader_train = DataLoader(dataset_train, batch_size=int(len(dataset_test)/34),shuffle=True,num_workers=4)
model = Classifier(num_classes=40)
model.load_state_dict(torch.load(model_path))
model.cuda()

model.eval()

for layer in model.bn_layers.extend(model.feature_extractor.bn_layers):
    layer.train()

correct_test=0
with torch.no_grad():
    for k, data_test in enumerate(dataset_loader_test):
        points = data_test[0].cuda()
        classifications = data_test[1].cuda()
        outputs = model(points)
        _ , output_classification = torch.max(outputs,dim=1)
        correct_test  += torch.sum((output_classification == classifications).float()).item()
    print(f"Accuracy test: {correct_test/len(dataset_test)}")

    correct_train=0
    for k, data_test in enumerate(dataset_loader_train):

      
        points = data_test[0].cuda()
        classifications = data_test[1].cuda()
        outputs = model(points)
        _ , output_classification = torch.max(outputs,dim=1)
        correct_train  += torch.sum((output_classification == classifications).float()).item()
    print(f"Accuracy train: {correct_test/len(dataset_test)}")
