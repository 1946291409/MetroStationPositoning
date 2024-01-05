'''
Descripttion: 对 模型使用数据 edge、Y这两个涉及id的进行重映射
Author: Xue
Date: 2023-11-22 14:35:42
LastEditTime: 2024-01-04 16:39:51
'''

import numpy as np

def reindex_Y():
    old_Y = np.load("/data/XueMengYang/MSP_new/XueProject/newdata/new/model_train_data/Y.npy")
    new_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/subway_re_ids.txt"
    id_map = dict()
    with open(new_path) as f:
        line = f.readline().strip()
        while line :
            data = line.split(',')
            id_map[data[0]] = int(data[1])
            line = f.readline().strip()
    id_map['-1'] = -1
    new_Y = []
    for i in old_Y:
        new_Y.append(id_map[i])
    # np.save("/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/model_train_data/re_Y.npy", new_Y)
reindex_Y()    
exit()
    
# reindex_Y()
# y = np.load("/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/model_train_data/re_Y.npy")
# print(y)
# print(y.shape)


def reindex_edges():
    old_edges = np.load("/data/XueMengYang/MSP_new/XueProject/newdata/new/model_train_data/edges.npy")
    new_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/base_re_ids.txt"
    print(old_edges.shape)
    # return
    id_map = dict()
    with open(new_path) as f:
        line = f.readline().strip()
        while line :
            data = line.split(',')
            id_map[data[0]] = int(data[1])
            line = f.readline().strip()
    new_edges = [[],[]]
    for i in range(len(old_edges[0])):
        new_edges[0].append(id_map[old_edges[0][i]])
        new_edges[1].append(id_map[old_edges[1][i]])
    np.save("/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/model_train_data/re_edges.npy", new_edges)

#reindex_edges()    
edges = np.load("/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/model_train_data/re_edges.npy")
print(edges.shape)

