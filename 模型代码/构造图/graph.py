'''
Descripttion: 生成gs、gt
Author: Xue
Date: 2023-11-22 22:01:55
LastEditTime: 2023-11-24 20:32:15
'''
import networkx as nx
import numpy as np 


def get_gs():
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/gs_edges_reindex.npy"
    data = np.load(path)
    # print(data)
    gs = nx.Graph()
    for i in range(337):
        gs.add_node(i)
    gs.add_edges_from(data) 
    print(gs)
    return gs
get_gs()
    
