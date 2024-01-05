'''
Description: 查看GCNII、GIN等模型的 匹配错误的点的连接情况
Author: Xue
Date: 2023-12-23 22:34:12
LastEditTime: 2023-12-23 22:36:57
'''
import numpy as np
import networkx as nx

path_GCNII = '/data/XueMengYang/MSP_new/XueProject/code/GNN_baselines/output/GCNII_12月23日15:39_0.9.npy'
path_GIN = "/data/XueMengYang/MSP_new/XueProject/code/GNN_baselines/output/GIN_12月23日05:33_0.91.npy"

# 加载数据
def load_data():
    gcnii_data = np.load(path_GCNII)
    gin_data = np.load(path_GIN)
    return gcnii_data, gin_data 

# 获取匹配错误的节点id
def get_err(data):
    err_x = []
    for arr in data:
        if arr[1] != arr[2]:
            err_x.append(arr[0])
    return err_x




def load_gb():
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/" + 'gb_edges_reindex.npy'
    data = np.load(path)
    # print(data)
    g = nx.Graph()
    g.add_weighted_edges_from(data)
    print(g)
    return g

