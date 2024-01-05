'''
Descripttion: GNN模型中 没有匹配正确的点的有什么特性，比如是否分散等 
Author: Xue
Date: 2023-11-23 10:19:27
LastEditTime: 2023-11-24 19:57:00
'''

import time
import networkx as nx
import numpy as np

# 加载 匹配错误数据
def get_error_data():
    # path = "/data/XueMengYang/MSP_new/XueProject/code/GNN_baselines/output/error.txt"
    # with open(path, 'r') as f:
    #     lines = f.readlines()
    # error_nodes = [int(node.strip()) for node in lines]
    path = "/data/XueMengYang/MSP_new/XueProject/code/GNN_baselines/output/error.npy"
    error_nodes = np.load(path)
    # print(error_nodes[0])
    return error_nodes[0]



# 生成基站图
def load_gb():
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/" + 'gb_edges_reindex.npy'
    data = np.load(path)
    # print(data)
    g = nx.Graph()
    g.add_weighted_edges_from(data)
    print(g)
    return g

# 检查是否聚集  => 聚集系数
def check():
    error_nodes = get_error_data()
    gb = load_gb()
    
    # 聚集系数
    subgraph = gb.subgraph(error_nodes)
    print(subgraph)
    clustering_coefficient = nx.average_clustering(gb)
    print(clustering_coefficient)
    
    # exit()
    # gb的连通分量
    # components = nx.connected_components(gb)
    # components = list(components)
    # components = sorted(components, key=lambda x: len(x), reverse=True)
    # for i in range(10):
    #     print(len(components[i]))
    # print(gb.number_of_nodes())
    
    len_err = len(error_nodes)
    print(len_err)
    not_connection = 0
    connection = 0
    dis = []
    import time
    start_time = time.time()
    for i in range(len_err):
        for j in range(i+1, len_err):
            if nx.has_path(gb,error_nodes[i], error_nodes[j]):
                connection += 1
                l = nx.shortest_path_length(gb, error_nodes[i], error_nodes[j])
                dis.append(l)
            else:
                not_connection += 1
        print(i, time.time()-start_time)
    np.save("/data/XueMengYang/MSP_new/XueProject/数据查验/未匹配正确的点/output/dis.npy",dis)
    print("connection: ", connection, "not_connection: ", not_connection,len_err*(len_err-1)/2)
    # for item in dis:
    #     print(item)

def draw():
    path = "/data/XueMengYang/MSP_new/XueProject/数据查验/未匹配正确的点/output/dis.npy"
    dis = np.load(path)
    print(max(dis))
    # return
    import matplotlib.pyplot as plt
    plt.hist(dis, bins=33)
    # print(time.time())
    plt.savefig("/data/XueMengYang/MSP_new/XueProject/数据查验/未匹配正确的点/output/" + "1.png" )


def check_connected_components():
    error_nodes = get_error_data()
    gb = load_gb()
    # gb的连通分量
    components = nx.connected_components(gb)
    components = list(components)
    components = sorted(components, key=lambda x: len(x), reverse=True)
    # for i in range(10):
    #     print(len(components[i]))
    print(gb.number_of_nodes())
    
    # error_node 有多少在 最大连通分量中
    max_comp = components[0]
    print(len(max_comp))
    count = 0
    for e in error_nodes:
        if e in max_comp:
            count += 1
    print(count,count/len(error_nodes))
    
    # 连通子图直径
    # max_subgraph = gb.subgraph(max_comp)
    # diameter = nx.diameter(max_subgraph)
    # print("最大连通分量的子图直径为",diameter)
    
    
    
    

def main():
    # check()
    # draw()
    check_connected_components()
    

if __name__ == "__main__":
    main()