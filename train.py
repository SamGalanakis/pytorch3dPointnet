from data_loader import ModelNet
from torch.utils.data import DataLoader
from model import Classifier
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm 


dataset = ModelNet(r'data/ModelNet40')

batch_size = 32
dataset_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)

model = Classifier(in_channels=3,feature_size=1024,num_classes=len(dataset.class_indexer),dropout=0)
model.cuda()
model.train()
optimizer = Adam(model.parameters(),lr=0.0001)
criterion = nn.CrossEntropyLoss()
loss_his = []
for epoch in tqdm(range(200)):
    for i, data in enumerate(dataset_loader):
        points = data[0].cuda()
        classifications = data[1].cuda()
        optimizer.zero_grad()
        

        outputs = model(points)
        loss = criterion(outputs,classifications) 
        loss.backward()
        optimizer.step()
        loss_his.append(loss.item()) 
        if i % 50 == 0:
            print(f"Loss of batch {i} is {loss.item()}")
model.sta('models')
            

        
