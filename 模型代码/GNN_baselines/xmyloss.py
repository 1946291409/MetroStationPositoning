'''
Descripttion: 损失函数的实现 FlocalLoss 和
Author: Xue
Date: 2023-11-02 20:24:02
LastEditTime: 2023-11-02 20:29:40
'''

import torch
from torch import nn

import torch
import torch.nn as nn

class FocalLoss(nn.Module):
   def __init__(self, alpha=0.25, gamma=2):
       super(FocalLoss, self).__init__()
       self.alpha = alpha
       self.gamma = gamma

   def forward(self, pred, target):
       num_classes = pred.size(1)
       one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
       pt = torch.where(one_hot > 0, pred, 1.0 - pred)
       loss = -self.alpha * (1.0 - pt) ** self.gamma * one_hot * torch.log(pt + 1e-8)
       return loss.sum() / (target != 0).sum()



class QFocalLoss(nn.Module):
   def __init__(self, alpha=0.25, gamma=2):
       super(QFocalLoss, self).__init__()
       self.alpha = alpha
       self.gamma = gamma

   def forward(self, pred, target):
       num_classes = pred.size(1)
       one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
       pt = torch.where(one_hot > 0, pred, 1.0 - pred)
       loss = -self.alpha * (1.0 - pt) ** self.gamma * one_hot * torch.log(pt + 1e-8)
       return loss.sum() / (target != 0).sum()