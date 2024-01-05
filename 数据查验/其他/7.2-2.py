"""
一个地铁站有多个基站，  那这些基站的连接情况？
"""
import networkx as nx
import matplotlib.pyplot as plt

pseudo_labels_path = "../data/beijing-原始的数据/data.txt"
ground_truth_path = "../data/beijing-原始的数据/bj_collect.txt"


# 读取伪标签数据
def load_pseudo_labels():
    subway_base = dict()  # key:地铁站   value；以该地铁站为伪标签的基站list
    with open(pseudo_labels_path, 'r') as f:
        line = f.readline().strip()
        while line:
            data = line.split()
            if data[-2] != '-1':
                key = data[-2]
                if key not in subway_base:
                    subway_base[key] = []
                subway_base[key].append(data[0])
            line = f.readline().strip()
    return subway_base


# 读取groundtruth数据
def load_ground_truth():
    subway_base = dict()  # key:地铁站   value；以该地铁站为伪标签的基站数
    with open(ground_truth_path, 'r') as f:
        line = f.readline().strip()
        while line:
            data = line.split(',')
            key = data[-1]
            if key not in subway_base:
                subway_base[key] = []
            subway_base[key].append(data[0])
            line = f.readline().strip()
    return subway_base


# 基站连接图
def generate_Gb():
    base_station_path = "../data/graph.txt"
    edges = []
    with open(base_station_path, 'r') as f:
        line = f.readline().strip()
        while line:
            data = line.split()
            edges.append([data[0], data[1]])
            line = f.readline().strip()
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


# 地铁站连接的基站情况
def judge(subway_base: dict):
    count_all = 0
    count_connected = 0
    G = generate_Gb()
    for bases in subway_base.values():
        #print(bases)
        print(len(bases))
        for i in range(len(bases)):
            if not G.has_node(bases[i]):
                continue
            for j in range(i + 1, len(bases)):
                if not G.has_node(bases[j]):
                    continue
                if nx.has_path(G, bases[i], bases[j]):
                    count_connected += 1
                count_all += 1
    with open("result.txt",'w') as f:
        f.write("基站连通率：{}".format(count_connected / count_all))
    #print("基站连通率：", count_connected / count_all)


if __name__ == '__main__':
    judge(load_pseudo_labels())

    #judge(load_ground_truth())
