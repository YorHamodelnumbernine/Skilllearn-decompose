# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 00:58:32 2021

@author: 陈博华
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResClassifier(nn.Module):
    def __init__(self, class_num, feature_dim, bottleneck_dim=256):
        super(ResClassifier, self).__init__()
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.bottleneck(x)
        y = self.fc(x)
        return x,y