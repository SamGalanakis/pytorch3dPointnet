from data_loader import ModelNet
from torch.utils.data import DataLoader
from model import Classifier
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm 
import wandb
import torch
import os 
from torch.optim.lr_scheduler import LambdaLR
wandb.init(project="pytorch3dpointnet")
dataset = ModelNet(r'data/ModelNet40')

batch_size = 32
dataset_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True,num_workers=4)

model = Classifier(in_channels=3,feature_size=1024,num_classes=len(dataset.class_indexer),dropout=0)
model.cuda()

#Must put model on cuda before making scheduler !

wandb.watch(model)
model.train()
optimizer = Adam(model.parameters(),lr=0.001)
#Divide learning rate by 2 at 20 epoch intervals
lamda_scheduler = lambda epoch: 1 / (2**(epoch//20))
scheduler = LambdaLR(optimizer, lr_lambda=lamda_scheduler)
criterion = nn.CrossEntropyLoss()

for epoch in tqdm(range(1000)):
    accuracy_epoch = 0
    loss_his = []
    
    for i, data in enumerate(dataset_loader):
        points = data[0].cuda()
        classifications = data[1].cuda()
        optimizer.zero_grad()
        

        outputs = model(points)
        
        loss = criterion(outputs,classifications) 
        loss.backward()
        optimizer.step()
        loss_his.append(loss.item()) 
        
        output_classification = torch.argmax(outputs,dim=1)
        correct  = torch.sum((output_classification == classifications).float()).item()
        accuracy = correct/len(output_classification)
        accuracy_epoch += (1/(i+1))*(accuracy-accuracy_epoch)

        print(f"Batch {i} of epoch {epoch}")




        
        
    
    scheduler.step(epoch)
    wandb.log({"Epoch accuracy": accuracy_epoch, "Train Loss": sum(loss_his)/len(loss_his)})
    
    if epoch != 0 and epoch % 50 == 0:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'{epoch}model.pt'))