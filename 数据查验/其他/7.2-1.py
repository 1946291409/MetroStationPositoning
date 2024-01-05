"""
已有标签中，地铁站拥有基站的数量 的分布直方图？
"""
import pickle

import networkx as nx
import matplotlib.pyplot as plt

pseudo_labels_path = "../data/beijing-原始的数据/data.txt"
ground_truth_path = "../data/beijing-原始的数据/bj_collect.txt"
path_450 = "../data/bj_station_trans_dict_2021.pkl"

def get_all_subways():
    subways = set()
    with open(path_450, 'rb') as f:
        data = pickle.load(f)
    for key, values in data.items():
        for value in values:
            subways.add(value)
    return subways

# todo 伪标签
def load_pseudo_labels():
    subway_base = dict()   # key:地铁站   value；以该地铁站为伪标签的基站数
    with open(pseudo_labels_path, 'r') as f:
        line = f.readline().strip()
        while line:
            data = line.split()
            if data[-2] != '-1':
                key = data[-2]
                if key not in subway_base:
                    subway_base[key] = 0
                subway_base[key] += 1
            line = f.readline().strip()

    # print(subway_base)
    # print(len(subway_base.values()))
    # print(len(subway_base))
    # exit(0)

    subways = get_all_subways()
    count_zero = len(subways) - len(subway_base)
    print(count_zero)
    # 画直方图
    counts = list(subway_base.values())
    counts += [0]*count_zero
    plt.hist(counts, bins=range(1, max(counts)+1))
    plt.xlabel('number of base station')
    plt.ylabel('count')
    plt.show()


# todo ground truth 标签
def load_ground_truth_labels():
    subway_base = dict()  # key:地铁站   value；以该地铁站为伪标签的基站数
    with open(ground_truth_path, 'r') as f:
        line = f.readline().strip()
        while line:
            data = line.split(',')
            key = data[-1]
            if key not in subway_base:
                subway_base[key] = 0
            subway_base[key] += 1
            line = f.readline().strip()
    # print(subway_base)
    # print(len(subway_base.values()))
    # print(len(subway_base))
    # exit(0)
    subways = get_all_subways()
    count_zero = len(subways) - len(subway_base)
    print(count_zero)
    # 画直方图
    counts = list(subway_base.values())
    counts += [0] * count_zero
    print(counts)
    plt.hist(counts, bins=range(1, max(counts) + 1))
    plt.xlabel('number of base station')
    plt.ylabel('count')
    plt.show()


if __name__ == '__main__':
    #load_pseudo_labels()
    load_ground_truth_labels()

