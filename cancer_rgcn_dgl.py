#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import numpy as np
from tqdm import tqdm
import rdflib as rl
import pickle
import torch
# import torchtuples as tt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
# from pycox.models import CoxPH, LogisticHazard
# from torch_geometric.data import Data, DataListLoader, DataLoader
# from torch_geometric.nn import RGCNConv, SAGEConv, GraphConv, SAGPooling, GCNConv
# from torch_geometric.nn import global_max_pool as gmp
from torch.utils.data import DataLoader
from rdflib import Graph
from sklearn import metrics
import numpy as np
from functools import partial
import torchvision
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torchvision.transforms as transforms
from torch.utils.data import random_split, IterableDataset

import dgl
import dgl.nn as dglnn
# In[2]:


import pickle

f = open('../GCN_training/ei.pkl','rb')
ei = pickle.load(f)
f.close()

f = open('../GCN_training/ei_attr.pkl','rb')
ei_attr = pickle.load(f)
f.close()
print(ei_attr[:10])

f = open('../GCN_training/seen.pkl','rb')
seen = pickle.load(f)
f.close()

# In[3]:


def exp_data(fname):
    f=open(fname)
    line=f.readlines()
    f.close()
    line=line[1:]
    ########
    output=[[0,0] for j in range(len(seen)+1)]
    for l in line:
        prot,exp_value=l.split('\t')
        exp_value=float(exp_value)
#         print(prot)
        if prot in seen:
#             print(prot)
#             print('Im there')
            output[seen[prot]][0]=exp_value
    return output


# In[4]:


def metabo_data(fname, output):
    f=open(fname)
    line=f.readlines()
    f.close()
    line=line[1:]
    ########
    for l in line:
        meta,meta_value=l.split('\t')
        meta_value=float(meta_value)
        if meta in seen:
            output[seen[meta]][1]=meta_value
    return output


# In[5]:

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label = []
feat_vecs=[]
f=open('../testingset/data/BRCA_label.txt')
lines=f.readlines()
f.close()
lines=lines[1:]
for l in tqdm(lines):
    try:
        l=l.split('\t')
        if os.path.isfile('../testingset/data/'+l[0]):
            temp=l[1]
            label.append(temp)
            exp_feature=exp_data('../testingset/data/'+l[0])
            metbo_feature = exp_feature
            feat_vecs.append(metbo_feature)
            #print(len(feat_vecs[0]))
#             print('../value_for_node/'+l[0])
#             print('../value_for_node/'+l[1])
            #print('len of feature',len(feat_vecs))
            #print('len of label',len(label))
        else:
            pass
            #print('Not exist!')
        
    except:
        pass
        #print('There is a problem!!!!')


# In[6]:


labels = [float(i) for i in label]


# In[7]:


dic={}
i=0
for e in ei_attr:
    if e[0] not in dic:
        dic[e[0]]=i
        i+=1

# In[17]:
u = torch.tensor(ei[0], dtype=torch.long).to(device)
v = torch.tensor(ei[1], dtype=torch.long).to(device)
g = dgl.graph((u, v)).to(device)

i = 0
etypes = [dic[a[0]] for a in ei_attr]
etypes = torch.tensor(etypes).to(device)


def load_data():
    dataset=[]    
    for e in range(len(feat_vecs)):
        x=torch.tensor(feat_vecs[e],dtype=torch.float)
        labell = labels[e]
        y = torch.tensor([labell])
        dataset.append((x, y))

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainset = MyDataset(trainset)
    testset = MyDataset(testset)
    return trainset, testset


# In[9]:


# train_loader=DataLoader(train_dataset)
# test_loader=DataLoader(test_dataset)


# In[10]:

class MyDataset(IterableDataset):

    def __init__(self, data):
        self.data = data
        
    def get_data(self):
        for x, y in self.data:
            yield (g, etypes, x, y)

    def __iter__(self):
        return self.get_data()
    
    def __len__(self):
        return len(self.data)
    
def get_batches(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    
def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    g, etypes, x, y = map(list, zip(*samples))
    graph_batch = dgl.batch(g)
    return graph_batch, th.stack(etypes), th.stack(x), th.stack(labels)

    

class MyNet(nn.Module):
    def __init__(self, l1=120, k=8, r=1, batch_size=2):
        super(MyNet, self).__init__()
#         self.conv1 = RGCNConv(k1,k2,27)
#         self.pool1 = SAGPooling(k2, ratio=r, GNN=GCNConv)
#         self.conv2 = GCNConv(k2,l1) 
#         self.fc1 = nn.Linear(l1,1)
        self.conv1 = dglnn.RelGraphConv(2, k, 27)
        gate_nn = th.nn.Linear(k, 1)  # the gate layer that maps node feature to scalar
        self.gap = dglnn.GlobalAttentionPooling(gate_nn)  # create a Global Attention Pooling layer
        self.batch_size = batch_size
        self.etypes = etypes



    def forward(self, g, data):
        x, y = data
        x = x.view(x.shape[1], -1)
        x = F.relu(self.conv1(g, x, self.etypes))
        x = self.gap(g, x)
        print(x.shape)
        return x


# In[11]:

def testf(loader,model):
    with torch.no_grad():
        model.eval()
        correct = 0
        for data in loader:
            x,y = data
            #data = data.to(device)
            output = model(data)
            output = output[0]
#             print(output)
            output = float(output>0.5)
#             print(output,data.y)
            if output == y:
#                 print('here')
                correct = correct + 1
#             print(correct)
        return float(correct) / float(len(loader))


def tune_train(config):
    maxacc=0
    best_model=0
    model=nn.DataParallel(MyNet(config["l1"], config["k"], config["r"], config["batch_size"]))
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)

    train_dataset, test_dataset=load_data()
    
    loader=get_batches(train_dataset,batch_size=config["batch_size"])
    test_loader=get_batches(test_dataset,batch_size=config["batch_size"])
    acc=[]
    
    for epoch in range(100):
        test_acc = testf(test_loader,model)
        print('before: ',test_acc,maxacc)
        model.train()
        loss_all = 0
        it = 0
        loss_=torch.nn.BCELoss()
        for data in loader:
            x, y = data
            optimizer.zero_grad()
            output = model(data)
            it+=1
            #y = torch.cat([y for x,y in data]).to(output.device)
            y = y.to(output.device)
            loss = loss_(output, y.view(output.shape[0],1))
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
#         print(output,y.view(output.shape[0],1))
        train_auc = testf(loader,model)
        test_auc = testf(test_loader,model)
#         print('Epoch: {:03d}, Loss: {:.5f}, Train AUC: {:.5f}, Test AUC: {:.5f}'.
#               format(epoch, loss_all, train_auc, test_auc))
        tune.report(loss=loss_all,accuracy=test_auc)
        if test_auc>maxacc:
            maxacc=test_auc
            best_model=model
#             print('best so far: ',test_auc)
            torch.save(best_model.module.state_dict(), "model_best.pth")
        if len(acc)==2:
            if test_auc<=acc[0] and test_auc<=acc[1]:
                continue
            else:
                del(acc[0])
                acc.append(test_auc)
        elif len(acc)<2:
            acc.append(test_auc)
        torch.save(model.module.state_dict(), "model_best.pth")
        print(maxacc, config)
preds=[]
trues=[]


def main():
#     config = {
#         "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
#         "k1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 6)),
#         "k2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 6)),
#         "r": tune.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
#         "lr": tune.loguniform(1e-4, 1e-1),
#         "batch_size": tune.choice([2, 4, 8, 16])
#     }
    l1 = [4, 8, 16, 32, 64, 128, 256]
    k = [4, 8, 16, 32, 64]
    r = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lr = [0.1, 0.3, 0.01, 0.03, 0.001, 0.003]
    batch_size = [2, 4, 6, 8, 16]
    for i in l1:
        for j in k:
            for a in r:
                for c in lr:
                    for b in batch_size:
                        tune_train({"l1": i, "k": j, "r":a, "lr": c, "batch_size": b})
                        return
#     scheduler = ASHAScheduler(
#             metric="loss",
#             mode="min",
#             max_t=40,
        
#             grace_period=1,
#             reduction_factor=2)
#     reporter = CLIReporter(
#             metric_columns=["loss", "accuracy", "training_iteration"])
#     result = tune.run(
#             tune_train,
#             resources_per_trial={"cpu": 1, "gpu": 1},
#             config=config,
#             scheduler=scheduler,
#             progress_reporter=reporter)

# In[ ]:


main()


# In[ ]:

