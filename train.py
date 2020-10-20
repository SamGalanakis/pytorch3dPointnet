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
import numpy as np
from scipy.interpolate import interp1d



wandb.init(project="pytorch3dpointnet")
 

dataset = ModelNet(r'data/ModelNet40',lazy=False)
dataset_test = ModelNet(r'data/ModelNet40',lazy=False,mode='test')

assert dataset.class_indexer == dataset_test.class_indexer, "Different dicts.. different labels!"
batch_size = 32
dataset_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True,num_workers=4)
batch_size_test = int(np.floor(len(dataset_test)/20)) #Otherwise too big for gpu mem ~ 19gigs needed for for oneshot.
dataset_loader_test = DataLoader(dataset_test, batch_size=batch_size_test,shuffle=True,num_workers=4)
#Keep ratio 0.7 so dropout 0.3 !
model = Classifier(in_channels=3,feature_size=1024,num_classes=len(dataset.class_indexer),dropout=0.3)
model.cuda()
 

 
#Must put model on cuda before making scheduler !

wandb.watch(model)
model.train()
optimizer = Adam(model.parameters(),lr=0.001)
#Divide learning rate by 2 at 20 epoch intervals
lamda_scheduler = lambda epoch: 1 / (2**(epoch//20))
scheduler = LambdaLR(optimizer, lr_lambda=lamda_scheduler)
#Scale momentum
max_momentum_epoch = 1000
batch_momentum_scheduler = lambda epoch: interp1d([0,max_momentum_epoch],[0.5,0.01])(min(epoch,max_momentum_epoch))
criterion = nn.CrossEntropyLoss()




for epoch in tqdm(range(100000)):
    #Set schedules momentum
    for bn_layer in model.bn_layers:
        bn_layer.momentum = batch_momentum_scheduler(epoch)
    accuracy_epoch = 0
    train_loss=0
    
    for i, data in enumerate(dataset_loader):
        points = data[0].cuda()
        classifications = data[1].cuda()
        optimizer.zero_grad()
        

        outputs = model(points)
        
        loss = criterion(outputs,classifications) 
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        
        output_classification = torch.argmax(outputs,dim=1)
        correct  = torch.sum((output_classification == classifications).float()).item()
        accuracy = correct/len(output_classification)
        accuracy_epoch += (1/(i+1))*(accuracy-accuracy_epoch)

        print(f"Batch {i} of epoch {epoch}")


    scheduler.step()

    
    if epoch != 0 and epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'{epoch}model.pt'))


    model.eval()
    test_loss=0
    with torch.no_grad():
        correct_test=0
        for k, data_test in enumerate(dataset_loader_test):
            points = data_test[0].cuda()
            classifications = data_test[1].cuda()
            outputs = model(points)
            test_loss += criterion(outputs,classifications).item()
            output_classification = torch.argmax(outputs,dim=1)
            correct_test  += torch.sum((output_classification == classifications).float()).item()
        accuracy_test = correct_test/len(dataset_test)
            
    model.train()
    wandb.log({
    "Train accuracy": accuracy_epoch,
    "Train oss": train_loss/len(dataset_loader),
    'Test accuracy':accuracy_test,
    'Test loss': test_loss/len(dataset_loader_test)
    
    }
    
    )
        
            






   