#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import namedtuple
from IPython.display import Image
from torch.utils import data
import torchvision   
#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
cuda = torch.cuda.is_available()
cuda
#from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


trainset = torchvision.datasets.ImageFolder(root='./11-785hw2p2-s20/train_data/medium/', 
                                                       transform=torchvision.transforms.ToTensor())
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transformtrain = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,])
transformval = transforms.Compose([
        transforms.ToTensor(),
        normalize,])
valset = torchvision.datasets.ImageFolder(root='./11-785hw2p2-s20/validation_classification/medium/', 
                                                       transform=transformval)
#valset = torchvision.datasets.ImageFolder(root='./11-785hw2p2-s20/validation_classification/medium/', 
#                                                       transform=torchvision.transforms.ToTensor())
#testset = torchvision.datasets.ImageFolder(root='./11-785hw2p2-s20/test_classification/medium/', 
#                                                       transform=torchvision.transforms.ToTensor())

print(len(trainset.classes), len(trainset))
print(len(valset.classes), len(valset))


# In[3]:


num_workers = 8 if cuda else 0 
train_loader_args = dict(shuffle=True, batch_size=128, num_workers=num_workers, pin_memory=True) if cuda  else dict(shuffle=True, batch_size=128)
train_loader = data.DataLoader(trainset, **train_loader_args)

val_loader_args = dict(shuffle=True, batch_size=128, num_workers=num_workers, pin_memory=True) if cuda  else dict(shuffle=True, batch_size=128)
val_loader = data.DataLoader(valset, **val_loader_args)


# In[4]:


import pandas as pd
df = pd.read_table("./11-785hw2p2-s20/test_trials_verification_student.txt", header=None, delimiter=" ")
print(df.head())


class VeriDataset(Dataset):
    def __init__(self, df, path):
        self.df = df
        self.path = path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img0 = Image.open(self.path+self.df.iloc[index,0])
        img1 = Image.open(self.path+self.df.iloc[index,1])
        img0 = transformval(img0)
        img1 = transformval(img1)
        return img0, img1
def parse_data(datadir):
    img_list = []
    for root, directories, filenames in os.walk(datadir):  #root: median/1
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
            ID_list.append(filename)
    print('{}\t\t{}\n'.format('#Images', len(img_list)))
    return img_list

path = ('./11-785hw2p2-s20/test_verification/')
veriset = VeriDataset(df, path)

veriloader = DataLoader(veriset, batch_size=128, shuffle=False, pin_memory=True, num_workers=1, drop_last=False)
print(next(iter(veriloader))[1].shape)
#print(len(next(iter(veriloader))))


# In[5]:



class ImageDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        #self.target_list = target_list
        #self.n_class = len(list(set(target_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = transformval(img)
        #label = self.target_list[index]
        return img#, label
ID_list = []


img_list = parse_data('./11-785hw2p2-s20/test_classification/medium')
testset = ImageDataset(img_list)
testloader = DataLoader(testset, batch_size=128, shuffle=False, pin_memory=True, num_workers=1, drop_last=False)

#veri_list = parse_data('./11-785hw2p2-s20/test_verification')
#veriset = ImageDataset(veri_list)
#veriloader = DataLoader(veriset, batch_size=128, shuffle=False, pin_memory=True, num_workers=1, drop_last=False)


# In[6]:


def predict(model, test_loader):
    model.eval()
    labelout = []
    for batch_num, feats in enumerate(test_loader):
        feats = feats.to(device)
        feature, outputs,_ = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        pred_temp = pred_labels.tolist()
        for i in range(len(pred_temp)):
            #print(int(trainset.classes[pred_temp[i]]))
            labelout +=[int(trainset.classes[pred_temp[i]])]
    return labelout

num_classes = len(trainset.classes)
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.resnet_conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.resnet_conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.resnet_conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.resnet_conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.resnet_conv5 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        
        self.shortcut1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(128)
        )
        self.resnet_conv6 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.resnet_conv7 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.resnet_conv8 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.resnet_conv9 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(128)
        
        self.resnet_conv10 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn10 = nn.BatchNorm2d(256)
        self.resnet_conv11 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.shortcut2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(256)
        )
        self.resnet_conv12 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(256)
        self.resnet_conv13 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn13 = nn.BatchNorm2d(256)
        
        self.resnet_conv14 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn14 = nn.BatchNorm2d(512)
        self.resnet_conv15 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn15 = nn.BatchNorm2d(512)
        self.shortcut3 = nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(512)
        )
        self.resnet_conv16 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(512)
        self.resnet_conv17 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn17 = nn.BatchNorm2d(512)
        
        self.linear = nn.Linear(2048,2300)
        
        self.linear_closs = nn.Linear(2048, 10, bias=True)
        self.relu_closs = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.resnet_conv1(x)))
        
        out = F.relu(self.bn2(self.resnet_conv2(out)))
        out = F.relu(self.bn3(self.resnet_conv3(out)))
        out = F.relu(self.bn4(self.resnet_conv4(out)))
        out = F.relu(self.bn5(self.resnet_conv5(out)))
        
        temp = out
        out = F.relu(self.bn6(self.resnet_conv6(out)))
        out = F.relu(self.bn7(self.resnet_conv7(out)) + self.shortcut1(temp))
        out = F.relu(self.bn8(self.resnet_conv8(out)))
        out = F.relu(self.bn9(self.resnet_conv9(out)))
 
        temp = out
        out = F.relu(self.bn10(self.resnet_conv10(out)))
        out = F.relu(self.bn11(self.resnet_conv11(out)) + self.shortcut2(temp))
        out = F.relu(self.bn12(self.resnet_conv12(out)))
        out = F.relu(self.bn13(self.resnet_conv13(out)))
        
        temp = out
        out = F.relu(self.bn14(self.resnet_conv14(out)))
        out = F.relu(self.bn15(self.resnet_conv15(out)) + self.shortcut3(temp))
        out = F.relu(self.bn16(self.resnet_conv16(out)))
        out = F.relu(self.bn17(self.resnet_conv17(out)))
        #print(out.shape)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #out = self.linear(out)
        #print(out.shape)
        label_output = self.linear(out)
        label_output = label_output/torch.norm(self.linear.weight, dim=1)
        closs_output = self.linear_closs(out)
        closs_output = self.relu_closs(closs_output)
        return closs_output, label_output, out

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


# In[7]:


class CenterLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, device=torch.device('cpu')):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) +                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss



def train_closs(model, data_loader, test_loader, task='Classification'):
    model.train()
    highest_acc = 0
    for epoch in range(numEpochs):
        avg_loss = 0.0
        pbar = tqdm(total=round(len(trainset)/128))
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer_label.zero_grad()
            optimizer_closs.zero_grad()
            #print(feats, len(model(feats)))
            
            feature, outputs,  = model(feats)

            l_loss = criterion_label(outputs, labels.long())
            c_loss = criterion_closs(feature, labels.long())
            loss = l_loss + closs_weight * c_loss
            
            loss.backward()
            
            optimizer_label.step()
            # by doing so, weight_cent would not impact on the learning of centers
            for param in criterion_closs.parameters():
                param.grad.data *= (1. / closs_weight)
            optimizer_closs.step()
            
            avg_loss += loss.item()

            if batch_num % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/50))
                avg_loss = 0.0    
            
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
            #pbar.update(1)
        pbar.close()
        print("this is epoch: "+ str(epoch+1))
        if task == 'Classification':
            val_loss, val_acc = test_classify_closs(model, test_loader)
            #train_loss, train_acc = test_classify_closs(model, data_loader)
            print('Val Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(val_loss, val_acc))
        else:
            test_verify(model, test_loader)


def test_classify_closs(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        feature, outputs, embedding = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        l_loss = criterion_label(outputs, labels.long())
        c_loss = criterion_closs(feature, labels.long())
        loss = l_loss + closs_weight * c_loss
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total


# In[ ]:





# In[8]:


device = torch.device("cuda" if cuda else "cpu")

numEpochs = 80
num_feats = 3

#learningRate = 1e-2
learningRate = 1e-2
weightDecay = 5e-5
closs_weight = 0.003
lr_cent = 0.5
feat_dim = 10


network = Network()
network.load_state_dict(torch.load('mar13_network_model.pt').state_dict())

criterion_label = nn.CrossEntropyLoss()
criterion_closs = CenterLoss(2300, feat_dim, device)
optimizer_label = torch.optim.SGD(network.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)
#optimizer_label = torch.optim.Adam(network.parameters(), lr=0.0002)
optimizer_closs = torch.optim.SGD(criterion_closs.parameters(), lr=lr_cent)

scheduler_label = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_label, patience=3)
scheduler_closs = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_closs, patience=3)

network.train()
network.to(device)


# In[9]:


val_loss, val_acc = test_classify_closs(network, val_loader)
            #train_loss, train_acc = test_classify_closs(model, data_loader)
print('Val Loss: {:.4f}\tVal Accuracy: {:.4f}'.
format(val_loss, val_acc))


# In[10]:


labelout = predict(network, testloader)
df = pd.DataFrame()
df['ID'] = ID_list
df['Category'] = labelout#.cpu().numpy()
df['IDnum'] = df.apply(lambda row: int(row['ID'][:4]) , axis=1)
df = df.sort_values(by=['IDnum'])
df = df.drop(['IDnum'], axis=1)
name = 'after data augmentation.csv'
df.to_csv(name, index=False)


# In[11]:


def verification(model, test_loader):
    model.eval()
    verout = []
    cos = nn.CosineSimilarity()
    for index, batch_num in enumerate(test_loader):
        #print(batch_num[0].shape, batch_num[1].shape)
        batch_num0,batch_num1 = batch_num
        batch_num0, batch_num1 = batch_num0.to(device),  batch_num1.to(device)
        _, _, embedding0 = model(batch_num0)
        _, _, embedding1 = model(batch_num1)
        if index%100==0:
            print("batch at: ", index)

        output = cos(embedding0,embedding1)
        #print(output.tolist())
        verout += output.tolist()
        del batch_num0
        del batch_num1
        del embedding0
        del embedding1
        del output
        
            #tempout = cosine_similarity(np.array(out0[i]).reshape(-1, 1), np.array(out1[i]).reshape(-1, 1))

    return verout


# In[12]:


embedding = verification(network, veriloader)


# In[13]:


print(len(embedding))


# In[14]:


print(embedding[:10])


# In[15]:


newdf = pd.read_table("./11-785hw2p2-s20/test_trials_verification_student.txt", names=['trial'])
newdf.head()


# In[16]:


newdf['score'] = embedding
name = "AUC_out.csv"
newdf.to_csv(name, index=False)


# In[ ]:





# In[ ]:




