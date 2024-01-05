"""
对450 和 337 站点的连通情况进行分析
"""
import matplotlib.pyplot as plt
import networkx as nx
import pickle

path_337 = "./data/station_seq.txt"
path_450 = "./data/bj_station_trans_dict_2021.pkl"


# plt.rcParams.update({
#     'figure.figsize':(8,6)
# })

def generateGraph():
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


# 先对 450 图进行构造
def load_450():
    G = generateGraph()
    # 检查连通分量
    connect_components = nx.connected_components(G)
    for component in connect_components:
        print(component)
    # 绘图
    pos = nx.spring_layout(G)  # 通过Spring layout算法计算节点位置
    plt.figure(figsize=(8, 8))
    # print(G.number_of_nodes())
    nx.draw_networkx(G, with_labels=False, pos=pos, node_size=200, alpha=0.8)
    plt.show()


# todo 筛选 337 的图
def load_337():
    with open(path_337, 'r') as f:
        lines = f.readlines()
    sub_nodes = [line.strip() for line in lines]
    G = generateGraph()
    sub_graph = G.subgraph(sub_nodes)

    # print(sub_graph.nodes())
    # exit(0)

    # 检查连通分量
    connect_components = nx.connected_components(sub_graph)
    for component in connect_components:
        print(component)

    # 绘图
    pos = nx.spring_layout(sub_graph)  # 通过Spring layout算法计算节点位置
    plt.figure(figsize=(8, 8))
    # print(G.number_of_nodes())
    nx.draw_networkx(sub_graph, with_labels=False, pos=pos, node_size=200, alpha=0.8)
    plt.show()



if __name__ == '__main__':
    # load_450()
    load_337()
