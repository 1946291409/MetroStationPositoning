"""
cross在不care地铁站的连接时,过多，需要查一下这个地铁站之间的距离分布
"""
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


pseudo_labels_path = "../data/beijing-原始的数据/data.txt"
ground_truth_path = "../data/beijing-原始的数据/bj_collect.txt"
path_450 = "../data/bj_station_trans_dict_2021.pkl"
base_station_path = "../data/graph.txt" 

# 加载伪标签数据
def load_pseudo_labels():
    base2subway = dict()    # dict{基站：地铁站}
    with open(pseudo_labels_path,'r') as f:
        line = f.readline().strip()
        while line:
            data = line.split()
            if data[-2]!='-1':
                base2subway[data[0]] = data[-2]
            line = f.readline().strip()
    return base2subway


# 加载groundtruth数据
def load_groundtruth():
    base2subway = dict()
    with open(ground_truth_path, 'r') as f:
        line = f.readline().strip()
        while line:
            data = line.split(",")
            base2subway[data[0]] = data[1]
            line = f.readline().strip()
    return base2subway


# 基站图
def generate_Gb():
    edges = []
    with open(base_station_path, 'r') as f:
        line = f.readline().strip()
        while line:
            data = line.split()
            edges.append([data[0], data[1]])
            line = f.readline().strip()
    Gb = nx.Graph()
    Gb.add_edges_from(edges)  
    return Gb


# 地铁站图
def genenrate_Gs():
    edges = []
    with open(path_450, 'rb') as f:
        data = pickle.load(f)
    for key, values in data.items():
        for value in values:
            edge = [key, value]
            edges.append(edge)
    Gs = nx.Graph()
    Gs.add_edges_from(edges)
    # 去掉自环
    Gs.remove_edges_from(nx.selfloop_edges(Gs))
    return Gs

distance_path = []
def judge(base2subway):
    gs = genenrate_Gs()
    gb = generate_Gb()
    cross = 0
    internal = 0
    for edge in gb.edges():
        b1 = edge[0]
        b2 = edge[1]
        if (b1 not in base2subway) or (b2 not in base2subway):
            continue
        if base2subway[b1] == base2subway[b2]:
            internal+=1
        else:
            cross+=1
            s1 = base2subway[b1]
            s2 = base2subway[b2]
            if (not gs.has_node(s1)) or (not gs.has_node(s2)):
                continue
            distance = nx.shortest_path_length(gs, s1, s2)
            distance_path.append(distance)
    print("internal:{},cross:{}".format(internal,cross))

def sort():
    counter = Counter(distance_path)
    top_10 = counter.most_common(10)
    print(top_10)


if __name__=='__main__':
    judge(load_pseudo_labels())
    sort()
