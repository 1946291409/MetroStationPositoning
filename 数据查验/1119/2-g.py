'''
Descripttion: 加载Gs图数据，解释为新数据集的边那么少  =》 没有问题
Author: Xue
Date: 2023-11-19 20:18:54
LastEditTime: 2023-11-19 20:35:10
'''

# 加载raw的Gs
import pickle

from numpy import sort
path_raw = "/data/XueMengYang/MSP_new/XueProject/data/beijing-原始的数据/bj_station_trans_dict_2021.pkl"
with open(path_raw, 'rb') as file:
    Gs_data = pickle.load(file)
edges = set()
for key,values in Gs_data.items():
    for v in values:
        if v != key:
            edges.add((min(key,v),max(key,v)))
print(len(edges))
print(len(Gs_data))
print(len(Gs_data)/len(edges))

# 加载新数据集的Gs
path_new = "/data/XueMengYang/MSP_new/XueProject/dataset/data/subway_graph.txt"
edges = set()
nodes = set()
with open(path_new, 'r') as file:
    line = file.readline()
    while line:
        data = line.strip().split(',')
        data = sorted(data)
        edges.add((data[0],data[1]))
        nodes.add(data[0])
        nodes.add(data[1])
        line = file.readline()
print(len(edges))
print(len(nodes))   
print(len(nodes)/len(edges))

'''
541
450
0.8317929759704251
385
337
0.8753246753246753
'''