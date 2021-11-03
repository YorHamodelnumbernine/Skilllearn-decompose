# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 23:51:09 2021

@author: 陈博华
"""

import os
import argparse
import tqdm
from itertools import chain
from collections import OrderedDict
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, pdb
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix

x = torch.rand(2, 3)
net1 = nn.Linear(3, 3)
net2 = nn.Linear(3, 3)
a = net1(x)
b = net2(a)

tgt = torch.rand(2, 3)
loss_fun = torch.nn.MSELoss()
opt1 = torch.optim.Adam(net1.parameters(), 0.002)
opt2 = torch.optim.Adam(net2.parameters(), 0.002)

for i in range(100):
    tmp = net1(x)
    output = net2(tmp)
    loss = loss_fun(output, tgt)

    net1.zero_grad()
    net2.zero_grad()

    loss.backward()
    opt1.step()
    opt2.step()

    print('EPOCH:{},loss={}'.format(i, loss))

aa = net1(x)
bb = net2(aa)

print(a)
print(aa)
print(b)
print(bb)