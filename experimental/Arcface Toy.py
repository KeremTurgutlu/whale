#!/usr/bin/env python
# coding: utf-8

# ArcFace paper: https://arxiv.org/pdf/1801.07698.pdf

# **Motivation :Good idea to craete toy Examples for Understandings Complex Concepts**

# ### Arcface Loss
# 
# Intra class compactness and Inter class discrepancy

# In[1]:


from fastai.core import *
from fastai.vision import *
from fastai.datasets import *


# In[2]:


get_ipython().system('pwd')


# In[3]:


path = untar_data(URLs.MNIST, dest="./data/mnist")


# In[4]:


path


# In[5]:


path = Path('../data/mnist/')


# In[6]:


path.ls()


# In[7]:


print(len((path/'training').ls()));(path/'training').ls()


# In[8]:


print(len((path/'testing').ls()));(path/'testing').ls()


# In[9]:


# delete classes 56789
# since we are forcing features to 2D space it's getting hard to discriminate many classes
del_class = ['5','6', '7','8','9']
for p in (path/'testing').ls():
    if p.name in del_class:
        shutil.rmtree(p)

for p in (path/'training').ls():
    if p.name in del_class:
        shutil.rmtree(p)


# In[10]:


tfms = get_transforms(do_flip=False)
testdata = (ImageItemList.from_folder(path/'testing'))        
data = (ImageItemList.from_folder(path/'training')    
    .random_split_by_pct(0.05)
    .label_from_folder()
    .add_test(testdata)         # only takes single 'label'      
    .transform(tfms, size=64)       
    .databunch()) 


# In[11]:


data


# In[12]:


Counter(data.train_ds.y.items)


# In[13]:


Counter(data.valid_ds.y.items)


# In[14]:


data.c


# ### Ipython Experiments

# In[15]:


from ipyexperiments import *


# In[16]:


with IPyExperiments():
    xb, yb = data.one_batch()
    data.show_batch()


# ### ArcLoss Model

# In[17]:


import pdb
from fastai.metrics import accuracy
from fastai.layers import *
from torch import nn
import torch.nn.functional as F
import torch
from fastai.callbacks import TerminateOnNaNCallback
from fastai.basic_data import DatasetType
from torch.nn.parameter import Parameter
from torch.nn import init


# In[18]:


class ArcLoss(nn.Module):
    def __init__(self, margin=0.5, scale=64):
        super().__init__()
        self.margin = margin
        self.scale = scale
        
    def forward(self, logits, target): 
        """
        - Normalize W and xi by l1 norm
        - Find original logit
        - Find theta for original target logit
        - Calculate marginal target logit
        - Calculate diff in marginal and original logit
        - Add it to original 
        - Scale logits
        - Calc cross entropy
        """
#         pdb.set_trace()        
        one_hot_gt = torch.eye(data.c)[target].byte().cuda()
        original_target_logits = logits.masked_select(one_hot_gt)
#         print(original_target_logits) 
        # -1 + eps to 1-eps with eps = 1e-7.
        # lifesaver: https://github.com/pytorch/pytorch/issues/8069
        thetas = torch.acos(original_target_logits.clamp_(-1+1e-7, 1-1e-7))
#         print(thetas)
        marginal_target_logits = torch.cos(thetas + self.margin)
        
        margin_target_diffs = marginal_target_logits - original_target_logits
        logits = logits + (one_hot_gt.float() * margin_target_diffs[:,None])
        logits = logits * self.scale
        probas = F.softmax(logits, dim=1)
#         print(probas)
        return F.cross_entropy(probas, target)
    
    def __repr__(self):
        return f"{self._get_name()} Margin : {self.margin}, Scale : {self.scale}"
    def __str__(self):
        return f"{self._get_name()} Margin : {self.margin}, Scale : {self.scale}"


# In[84]:


class FaceLoss(nn.Module):
    def __init__(self, m1=1, m2=0, m3=0, scale=64):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.scale = scale
        
    def forward(self, logits, target): 
        """
        - Normalize W and xi by l1 norm
        - Find original logit
        - Find theta for original target logit
        - Calculate marginal target logit
        - Calculate diff in marginal and original logit
        - Add it to original 
        - Scale logits
        - Calc cross entropy
        """
#         pdb.set_trace()        
#         print(logits[:2])
        one_hot_gt = torch.eye(data.c)[target].byte().cuda()
        original_target_logits = logits.masked_select(one_hot_gt)
#         print(original_target_logits) 
        # -1 + eps to 1-eps with eps = 1e-7.
        # lifesaver: https://github.com/pytorch/pytorch/issues/8069
        thetas = torch.acos(original_target_logits.clamp_(-1+1e-7, 1-1e-7))
#         print(thetas)
        marginal_target_logits = torch.cos(self.m1*thetas + self.m2) - self.m3
        
#         margin_target_diffs = marginal_target_logits - original_target_logits
        logits = self.scale * ((one_hot_gt.float()) * marginal_target_logits[:,None] + ((1 - one_hot_gt.float())) * logits)
#         print(logits[:2])
        probas = F.softmax(logits, dim=1)
#         print(probas)
        return F.cross_entropy(probas, target)
    
    def __repr__(self):
        return f"{self._get_name()} Margins : {(self.m1, self.m2, self.m3)}, Scale : {self.scale}"
    def __str__(self):
        return f"{self._get_name()} Margins : {(self.m1, self.m2, self.m3)}, Scale : {self.scale}"


# In[85]:


class ArcHead(nn.Module):
    def __init__(self, n_features=512, p=0.5):
        super().__init__()
        self.layers = nn.Sequential(*(
                        [AdaptiveConcatPool2d(), Flatten()]+\
                        bn_drop_lin(1024, 128, p=p, actn=nn.ReLU())+\
                        bn_drop_lin(128, n_features, p=p, actn=None)+\
                        [nn.BatchNorm1d(n_features),
                         nn.Linear(n_features, data.c, bias=False)]))
        
    def forward(self, x):    
        """bs x n_features"""
        # embedding feature
        x = self.layers[:-1](x); x = F.normalize(x)
        # class center weight 
        self.layers[-1].weight.data.div_(self.layers[-1].weight.norm(dim=0))
        logits = self.layers[-1](x)
#         print(logits)
        return logits


# In[86]:


custom_head = ArcHead(n_features=2)
learner = create_cnn(data, models.resnet18, custom_head=custom_head, metrics=[accuracy],
                     callbacks=[TerminateOnNaNCallback()], model_dir="../models/")
loss = FaceLoss(1, 0.25, 0,scale=12); print(loss) 
learner.loss_func = loss


# In[87]:


learner.model[1]


# In[48]:


# with IPyExperiments():
#     xb,yb = data.one_batch(ds_type=DatasetType.Train)
#     print(xb.shape, yb.shape)
#     out = learner.model(xb.cuda())
#     print(out.shape)
#     loss_ = loss(out, yb.cuda())
#     print(loss_)
#     acc = learner.metrics[0](out, yb.cuda())
#     print(acc)


# In[88]:


learner.lr_find()


# In[89]:


learner.recorder.plot()


# In[90]:


# arcloss
learner.fit_one_cycle(20, max_lr=1e-1, wd=1e-4)


# In[31]:


# softmax
learner.fit_one_cycle(5, max_lr=1e-1, wd=1e-4)


# In[ ]:





# ### Add hook to get features during inference

# In[91]:


from fastai.callbacks import hook_outputs
from fastai.basic_data import DatasetType
from fastai.torch_core import *


# In[92]:


[children(children(learner.model[1])[0])[-2]]


# In[93]:


# register a hook to grab features
hook = hook_outputs([children(children(learner.model[1])[0])[-2]])


# In[94]:


xb, yb = data.one_batch(ds_type=DatasetType.Valid)


# In[95]:


model = learner.model.eval()
outb = model(xb.cuda())


# In[96]:


outb.shape


# In[97]:


# normalize features to have norm ||x|| = 1
out_features = F.normalize(hook.stored[0])
out_features[0].norm().data.item()


# In[98]:


out_features = to_np(out_features.data)


# In[99]:


out_features.shape


# In[100]:


(out_features[0]**2).sum()


# ### Visualize Validation Data

# In[101]:


import seaborn as sns


# In[102]:


centers = F.normalize(children(children(learner.model[1])[0])[-1].weight.data)
centers = to_np(centers)


# In[103]:


colors = dict(zip(range(data.c),sns.color_palette()))
colors = {k:np.array(colors[k])[None] for k in colors}
colors


# #### ArcLoss

# In[104]:


# ARCLOSS
with IPyExperiments():
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    for i, (x,y) in enumerate(zip(centers[:,0], centers[:,1])):
        ax.scatter(x,y, c=colors[i])
        ax.plot((0,x), (0,y))
        ax.text(x,y,i,size=16)
    ax.scatter(0,0)
    ax.text(0,0,'origin',size=16)

    # add validation data
    for xb, yb in learner.data.valid_dl:
        outb = model(xb.cuda())
        out_features = F.normalize(hook.stored[0])
        for label, (x,y) in zip(to_np(yb), out_features):
            ax.scatter(x,y,c=colors[label])


# In[76]:


F.normalize(children(learner.model[1])[0][-1].weight)**2


# #### Softmax 

# In[45]:


# SOFTMAX
with IPyExperiments():
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    for i, (x,y) in enumerate(zip(centers[:,0], centers[:,1])):
        ax.scatter(x,y, c=colors[i])
        ax.plot((0,x), (0,y))
        ax.text(x,y,i,size=16)
    ax.scatter(0,0)
    ax.text(0,0,'origin',size=16)

    # add validation data
    for xb, yb in learner.data.valid_dl:
        outb = model(xb.cuda())
        out_features = F.normalize(hook.stored[0])
        for label, (x,y) in zip(to_np(yb), out_features):
            ax.scatter(x,y,c=colors[label])


# In[46]:


len(data.valid_ds)


# In[ ]:




