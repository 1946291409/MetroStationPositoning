'''
Description: 
Author: Xue
Date: 2023-12-21 17:03:50
LastEditTime: 2023-12-21 17:15:01
'''


# 实现GCN 导入的包
from os import path
from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, ChebConv, GINConv, MLP, GINEConv, GIN, GCN2Conv
import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm



class GAT_Net_test(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GAT_Net_test, self).__init__()
        # 设置网络参数
        self.emb_dim = in_feats
        self.out_dim = out_feats
        self.num_heads = 2  # 多头 
        self.dropout = 0.3
        self.depth = 2
        self.convs = nn.ModuleList()
        ins = [in_feats, in_feats*self.num_heads]
        outs = [in_feats,in_feats]  # 每层实际的输出是 outs[i]*heads
        
        # 定义网络结构 2*GAT + 1*line 
        for i in range(self.depth):
            if i == self.depth - 1: # 最后一层GAT ，则不进行concate拼接，而是选择mean聚合。
                self.convs.append(GATConv(
                ins[i],
                outs[i],
                heads=self.num_heads,
                edge_dim=1,
                dropout=self.dropout,
                concat=True
                ))
            else :
                self.convs.append(GATConv(
                    ins[i],
                    outs[i],
                    heads=self.num_heads,
                    edge_dim=1,
                    dropout=self.dropout
                ))
        self.fc = nn.Linear(outs[self.depth-1]*self.num_heads, out_feats)
        
        
    def forward(self, x, edge_index, edge_weight):
        if edge_weight != None:
            edge_weight = edge_weight.unsqueeze(-1) # （N,） -> （N,1）
        # # x = F.dropout(x, p=self.dropout, training=self.training)   
        # x = F.relu(self.conv1(x, edge_index, edge_weight)) 
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.conv2(x, edge_index, edge_weight)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x