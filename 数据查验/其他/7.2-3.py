"""
假如A B两个地铁站相连接，则他们匹配的基站中，有多少是连着的？
"""
import networkx as nx
import pickle

pseudo_labels_path = "../data/beijing-原始的数据/data.txt"
ground_truth_path = "../data/beijing-原始的数据/bj_collect.txt"
path_450 = "../data/bj_station_trans_dict_2021.pkl"


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


# 地铁站图
def genenrate_Gs():
    edges = []
    with open(path_450, 'rb') as f:
        data = pickle.load(f)
    for key, values in data.items():
        for value in values:
            edge = [key, value]
            edges.append(edge)
    # print(edges)
    # print(len(edges))
    G = nx.Graph()
    G.add_edges_from(edges)
    # 去掉自环
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


# 基站图
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


def judge(subway_base: dict):
    count_all = 0
    count_connected = 0
    Gs = genenrate_Gs()
    Gb = generate_Gb()
    for edge in Gs.edges(data=False):
        print(edge)
        s1 = edge[0]
        s2 = edge[1]
        if (s1 not in subway_base) or (s2 not in subway_base):
            continue
        for b1 in subway_base[s1]:
            if not Gb.has_node(b1):
                continue
            for b2 in subway_base[s2]:
                if (not Gb.has_node(b2)) or (b1 == b2):
                    continue
                if Gb.has_edge(b1, b2):
                    count_connected += 1
                count_all += 1
    with open("result-7.2-3_pseudo.txt", 'w') as f:
        f.write("不同地铁站之间，基站连接率：{},其中连接数:{}".format(count_connected / count_all,count_connected))


if __name__ == '__main__':
    judge(load_ground_truth())
