#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from fastai import *
from fastai.vision import *
from ipyexperiments import *


# In[3]:


data_path = Path('../data/whale/')


# In[8]:


class myOptimWrapper(OptimWrapper):
    def step(self):          pass
    def zero_grad(self):      pass
    def real_step(self):      super().step()
    def real_zero_grad(self): super().zero_grad()
        
    def on_epoch_end(self):
        self.real_step()
        self.real_zero_grad()


# In[9]:


def my_create_opt(self, lr:Floats, wd:Floats=0.)->None:
    "Create optimizer with `lr` learning rate and `wd` weight decay."
    self.opt = myOptimWrapper.create(self.opt_func, lr, self.layer_groups,
                                     wd=wd, true_wd=self.true_wd, bn_wd=self.bn_wd)


# In[10]:


Learner.create_opt = my_create_opt


# In[11]:


path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)


# In[12]:


learn = create_cnn(data, models.resnet18, metrics=accuracy)


# In[ ]:





# In[ ]:


Learner()


# In[5]:


opt = myOptimWrapper()


# In[ ]:




