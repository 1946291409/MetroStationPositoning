'''
Descripttion: 
Author: Xue
Date: 2023-11-22 20:32:58
LastEditTime: 2023-11-23 10:17:41
'''


import os

from torch import le
import numpy as np
import networkx as nx


def get_reindex_gs():
    gs_map = dict()
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/subway_re_ids.txt"
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        data = line.strip().split(",")
        gs_map[data[0]] = int(data[1])
    return gs_map
    
def get_reindex_gb():
    gt_map = dict()
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/base_re_ids.txt"
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        data = line.strip().split(",")
        gt_map[data[0]] = int(data[1])
    # print(len(gt_map))
    return gt_map 

def reindex_gt():
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new/gt_data.txt"
    gt_data = []
    gb_map = get_reindex_gb()
    gs_map = get_reindex_gs()
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        data = line.strip().split(",")
        gt_data.append([gb_map[data[0]],gs_map[data[1]]])
    save_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/gt_reindex.txt"
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            for line in gt_data:
                f.write(",".join(str(x) for x in line) + "\n")
        print("save gt okk")

# reindex_gt()

def reindex_gs_edges():
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new/gs_edges_data.txt"
    edges = []
    gs_map = get_reindex_gs()
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        data = line.strip().split(',')
        edges.append([gs_map[data[0]],gs_map[data[1]]])
    print(len(edges))
    
    save_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/" + 'gs_edges_reindex.npy'
    np.save(save_path, edges)

# reindex_gs_edges()

# path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/" + 'gs_edges_reindex.npy'
# data = np.load(path)
# print(data)
# g = nx.Graph()
# g.add_edges_from(data)
# print(g)

def reindex_gb_edges():
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new/gb_edges(weight)_data.txt"
    edges = []
    gb_map = get_reindex_gb()
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        data = line.strip().split(',')
        edges.append([gb_map[data[0]],gb_map[data[1]],float(data[2])])
    # print(len(edges))
    
    save_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/" + 'gb_edges_reindex.npy'
    np.save(save_path, edges)

# reindex_gb_edges()
# path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/" + 'gb_edges_reindex.npy'
# data = np.load(path)
# print(data)
# g = nx.Graph()
# g.add_weighted_edges_from(data)
# print(g)


