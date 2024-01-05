import math
import numpy as np
import networkx as nx
from sklearn.datasets import load_sample_image
from utils.common import timer


root_path = "/data/XueMengYang/MSP_new/XueProject/dataset/data/"

# TODO:加载基站图
def load_base_graph():
    Gb = nx.Graph()
    base_data_path = root_path + "base_graph.txt"
    with open(base_data_path, "r") as f:
        line = f.readline().strip()
        while line:
            data = line.split(",")
            Gb.add_edge(data[0], data[1])
            line = f.readline().strip()
    return Gb   # 基站图

# TODO 加载groundtruth 数据
def load_groundtruth():
    # 创建一个地铁站和基站映射的map {subway:[base1,base2,...]}
    subwayTobase = dict()
    path = root_path + "ground_truth.txt"
    with open(path, "r") as f:
        line = f.readline().strip()
        while line:
            data = line.split(",")
            if subwayTobase.get(data[1]) is None:
                subwayTobase[data[1]] = [data[0]]
            else:   
                subwayTobase[data[1]].append(data[0])
            line = f.readline().strip()
    return subwayTobase

# a 和b是否 k 跳可达
def is_kth_neighbor(graph, a, b, k):
    if k == 0:
        return a == b
    elif k < 0:
        return False
    
    for neighbor in graph.neighbors(b):
        if is_kth_neighbor(graph, a, neighbor, k - 1):
            return True
    
    return False

# TODO 检测：地铁站u 对应的基站{a,b,c}是否是连接紧密的
@timer
def check(subway_to_base:dict, Gb:nx.Graph):
    # 检测：地铁站u 对应的基站{a,b,c}是否是连接紧密的 (检测是否为邻居)
    for  k in [1,2,3,4,5]:
        for subway, base in subway_to_base.items():
            count_true = 0
            count_sum = 0
            for i in range(len(base)):
                count_sum += (len(base) * (len(base) - 1))/2
                for j in range(i+1, len(base)):
                    if is_kth_neighbor(Gb, base[i],base[j],k):
                        count_true+=1
        print("对于{}阶：检测：地铁站u 对应的基站是否是连接紧密的 (检测是否为邻居) 正确率为：{}".format(k,count_true/count_sum))



def main():
    Gb = load_base_graph()
    subway_to_base = load_groundtruth()
    check(subway_to_base, Gb)





if __name__ == "__main__":
    main()



