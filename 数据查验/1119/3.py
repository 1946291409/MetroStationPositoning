'''
Descripttion: 查验基站图edge数量，为啥记录是205302，但是txt中有258155行  =》 确实是205302，因为有一些重复的边
Author: Xue
Date: 2023-11-19 21:25:35
LastEditTime: 2023-11-19 21:39:40
'''

import networkx as nx
import numpy as np

def get_new_gb():
    path = "/data/XueMengYang/MSP_new/XueProject/dataset/Adj_data/Gb.npy"
    gs = nx.from_numpy_array(np.load(path))
    print(gs)
    # Graph with 39490 nodes and 205302 edges
    
def get_raw_gb():
    Gb = nx.Graph()
    path = "/data/XueMengYang/MSP_new/XueProject/dataset/data/base_graph.txt"
    edges = set()
    with open(path,'r') as f:
        line = f.readline().strip()
        while line:
            data = line.split(',')
            bs = sorted(data[:2])
            Gb.add_edge(bs[0],bs[1])
            edges.add(tuple(bs))
            line = f.readline().strip()
    print(len(edges))
    print(Gb)
            
            
get_raw_gb()