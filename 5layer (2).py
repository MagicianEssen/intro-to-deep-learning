#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import namedtuple
from IPython.display import Image
from torch.utils import data
#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
cuda = torch.cuda.is_available()
cuda


# In[24]:


#test = np.zeros(())


# In[4]:


train = np.load("./train.npy", allow_pickle=True)
test = np.load("./test.npy", allow_pickle=True)
train_labels = np.load("./train_labels.npy", allow_pickle=True)
dev = np.load("./dev.npy", allow_pickle=True)
dev_labels = np.load("./dev_labels.npy", allow_pickle=True)





class MyDataset(data.Dataset):
    def __init__(self, X, Y, ctx):
        X_num = 0
        for i in range(len(X)):
            X_num += len(X[i])
        X_data_tmp = np.zeros((X_num, 40))
        loc = 0
        for i in range(len(X)):
            X_data_tmp[loc:loc+len(X[i])] = X[i]
            loc += len(X[i])
        self.idx = np.arange(ctx, ctx+X_num)
        self.X = np.pad(X_data_tmp, ((ctx,ctx), (0,0)), 'reflect').astype(np.float32)              
        Y_data = []
        for i in range(len(Y)):
            Y_data = np.concatenate((Y_data, Y[i]), axis = 0)   
        self.Y = Y_data.astype(np.int64)  
        self.ctx = ctx
        #print(self.X)
        #print(self.Y)

        
    def __len__(self):
        return len(self.Y)

    def __getitem__(self,index):
        getloc = self.idx[index]
        getx = np.concatenate(self.X[getloc-self.ctx:getloc+self.ctx+1].astype(np.float32))
        gety = self.Y[index].astype(np.int64)  
        return getx, gety
    





num_workers = 8 if cuda else 0 

train_dataset = MyDataset(train, train_labels, 12)
train_loader_args = dict(shuffle=True, batch_size=2048, num_workers=num_workers, pin_memory=True) if cuda                    else dict(shuffle=True, batch_size=2048)
train_loader = data.DataLoader(train_dataset, **train_loader_args)

dev_dataset = MyDataset(dev, dev_labels, 12)
dev_loader_args = dict(shuffle=True, batch_size=2048, num_workers=num_workers, pin_memory=True) if cuda                   else dict(shuffle=False, batch_size=2048)
dev_loader = data.DataLoader(dev_dataset, **dev_loader_args)




class MyDatasetTest(data.Dataset):
    def __init__(self, X, ctx):
        X_num = 0
        for i in range(len(X)):
            X_num += len(X[i])
        X_data_tmp = np.zeros((X_num, 40))
        loc = 0
        for i in range(len(X)):
            X_data_tmp[loc:loc+len(X[i])] = X[i]
            loc += len(X[i])
        self.idx = np.arange(ctx, ctx+X_num)
        self.len = len(X_data_tmp)
        self.X = np.pad(X_data_tmp, ((ctx,ctx), (0,0)), 'reflect').astype(np.float32)              
        self.ctx = ctx
        
    def __len__(self):
        return self.len

    def __getitem__(self,index):
        getloc = self.idx[index]
        getx = self.X[getloc-self.ctx:getloc+self.ctx+1].astype(np.float32)
        
        return getx

test_dataset = MyDatasetTest(test, 12)
test_loader_args = dict(shuffle=False, batch_size=2048)
test_loader = data.DataLoader(test_dataset, **test_loader_args)





#class SpeechModel(nn.Module):
#    def __init__(self):
#        super(SpeechModel, self).__init__()
#        self.fc1 = nn.Linear(1000, 2048)
#        self.bn1 = nn.BatchNorm1d(num_features=2048)
#        self.fc2 = nn.Linear(2048, 4096)
#        self.bn2 = nn.BatchNorm1d(num_features=4096)
#        self.fc3 = nn.Linear(4096, 2048)
#        self.bn3 = nn.BatchNorm1d(num_features=2048)
#        self.fc4 = nn.Linear(2048, 1000)
#        self.bn4 = nn.BatchNorm1d(num_features=1000)
#        self.fc5 = nn.Linear(1000, 138)
#    
#    def forward(self, x):
#        x = F.relu(self.bn1(self.fc1(x)))
#        x = F.relu(self.bn2(self.fc2(x)))
#        x = F.relu(self.bn3(self.fc3(x)))
#        x = F.relu(self.bn4(self.fc4(x)))
#        x = F.log_softmax(self.fc5(x))
#        return x
    
class SpeechModel(nn.Module):
    def __init__(self):
        super(SpeechModel, self).__init__()
        self.fc1 = nn.Linear(1000, 2048)
        self.bn1 = nn.BatchNorm1d(num_features=2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(num_features=1024)
        self.fc3 = nn.Linear(1024, 138)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.log_softmax(self.fc3(x))
        return x
#print(SpeechModel())


# In[9]:


def predict(model, loader):
    correct = 0
    labelout = []

    for data in loader:
        X = Variable(data.view(-1, 1000))
        X = X.to(device)
        out = model(X)
#         labelout.append(out.data.max(1, keepdim=True).tolist()[1])
        _, predicted = torch.max(out.data, 1)
        labelout += predicted.tolist()
    print(labelout[:5])
    return labelout

def inference(model, loader):
    correct = 0
    test_size = 0
    for data, label in loader:
        X = Variable(data.view(-1, 1000))
        Y = Variable(label)
        X = X.to(device)
        Y = Y.to(device)
        out = model(X)
        pred = out.data.max(1, keepdim=True)[1]
        predicted = pred.eq(Y.data.view_as(pred))
        correct += predicted.sum()
        test_size += len(data)
        #print(correct.item(), test_size)
    #print(correct.item()/test_size)
    return correct.item() / test_size, test_size

class Trainer():
    """ 
    A simple training cradle
    """
    
    def __init__(self, model, optimizer, load_path=None):
        self.model = model
        if load_path is not None:
            self.model = torch.load(load_path)
        self.optimizer = optimizer
            
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def run(self, epochs):
        print("Start Training...")
        self.metrics = []
        for e in range(n_epochs):
            scheduler.step()
            epoch_loss = 0
            correct = 0
            train_size = 0
            pbar = tqdm(total=round(15388713/2048))
            for data, label in train_loader:
                self.optimizer.zero_grad()
                X = Variable(data.view(-1, 1000))
                Y = Variable(label)
                
                X = X.to(device)
                Y = Y.to(device)
                
                out = self.model(X)
                pred = out.data.max(1, keepdim=True)[1]
                predicted = pred.eq(Y.data.view_as(pred))
                correct += predicted.sum()
                loss = F.nll_loss(out, Y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                train_size += len(data)
                pbar.update(1)
            pbar.close()
            total_loss = epoch_loss/train_size
            train_error = 1.0 - correct/train_size
            print("epoch: {0}, loss: {1:.8f}".format(e+1, total_loss))
            
            adam_trainer.save_model('./adam_model{}.pt'.format(str(e+1)))
#             self.model.load_state_dict(torch.load('./adam_model.pt'))
            test_acc, test_size = inference(self.model, dev_loader)
            print(test_acc)
            print("Test accuracy of model optimizer with Adam: {}".format(test_acc * 100))
            label_out = predict(self.model, test_loader)
            finalout = []
            for i in range(len(label_out)):
                finalout.append(label_out[i])
            idout = list(np.arange(0,223592,1))
            name = ('scores_5layers_epoch'+ str(e+1)+ '.csv')
            np.savetxt(name, [p for p in zip(idout, finalout)], delimiter=',', fmt='%s')



# In[ ]:


### LET'S TRAIN ###

def init_xavier(m):
    if type(m) == nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0,std)
# A function to apply "normal" distribution on the parameters
def init_randn(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0,1)

# We first initialize a Fashion Object and initialize the parameters "normally".
normalmodel = SpeechModel()
normalmodel.apply(init_xavier)

device = torch.device("cuda" if cuda else "cpu")
normalmodel.to(device)

n_epochs = 25

print("ADAM OPTIMIZER")
AdamOptimizer = torch.optim.Adam(normalmodel.parameters(), lr=0.0002)
scheduler = torch.optim.lr_scheduler.StepLR(AdamOptimizer, step_size=4, gamma=0.1)
adam_trainer = Trainer(normalmodel, AdamOptimizer)
adam_trainer.run(n_epochs)
# adam_trainer.save_model('./adam_model.pt')
print('')


# In[ ]:





