"""
基站图的边数、节点数 与9w基站的差距
以及与 9w基站中有label的节点的差距
"""

import networkx as nx
import pickle

pseudo_labels_path = "../data/beijing-原始的数据/data.txt"
ground_truth_path = "../data/beijing-原始的数据/bj_collect.txt"
path_450 = "../data/bj_station_trans_dict_2021.pkl"

has_label_base = []
has_groundtruth_base = []
all_base = []
base_in_graph = []
pseudolabel_base = dict()
groundtruth_base = dict()

pseudolabel_subway = set()



# 读取伪标签数据
def load_pseudo_labels():
    with open(pseudo_labels_path, 'r') as f:
        line = f.readline().strip()
        while line:
            data = line.split()
            all_base.append(data[0])
            if data[-2] != '-1':
                pseudolabel_subway.add(data[-2])
                has_label_base.append(data[0])
                pseudolabel_base[data[0]] = data[-2]
            line = f.readline().strip()

# 读取groundtruth数据
def load_ground_truth():
    with open(ground_truth_path, 'r') as f:
        line = f.readline().strip()
        while line:
            data = line.split(',')
            groundtruth_base[data[0]] = data[1]
            has_groundtruth_base.append(data[0])
            line = f.readline().strip()

# 基站图
Gb = nx.Graph()
def generate_Gb():
    base_station_path = "../data/graph.txt"
    edges = []
    with open(base_station_path, 'r') as f:
        line = f.readline().strip()
        while line:
            data = line.split()
            edges.append([data[0], data[1]])
            line = f.readline().strip()
    Gb.add_edges_from(edges)
    base_in_graph.extend(list(Gb.nodes()))

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
    print(G.number_of_nodes())
    n1 = '110000799560604'
    n2 = '110000799560605'
    if G.has_node(n1):
        print("111111")
    if G.has_node(n2):
        print("22222")
    return G
genenrate_Gs()



load_ground_truth()
generate_Gb()
load_pseudo_labels()
print("伪标签中地铁站的种类：{}".format(len(pseudolabel_subway)))
# if '110000799560604' in pseudolabel_subway:
#     print("111-----")
# exit(0)
print(" 有标签的基站：{}\n groundtruth的基站:{}\n 所有的base:{} \n graph里的base:{}"\
      .format(len(has_label_base),len(has_groundtruth_base) ,len(all_base) ,len(base_in_graph)))

count_has = 0
count_right = 0
for k,v in pseudolabel_base.items():
    if k in groundtruth_base:
        count_has+=1
        g_s = groundtruth_base[k]
        if g_s == v:
            count_right+=1
print("伪标签中有{}个groundtruth中的基站,其中正确的有{}个".format(count_has,count_right))

# if(len(pseudolabel_base) == len(has_label_base) ):
#     print(" 相等的")
#     exit(0)

in_pseudolabel = 0
in_gt = 0
with_out = 0
both_have = 0
for node in Gb.nodes():
    f_label = False
    f_ground = False
    if node in has_label_base:
        f_label = True
    if node in has_groundtruth_base:
        f_ground = True
    if f_label:
        in_pseudolabel+=1
    if f_ground:
        in_gt+=1
    if (not f_label) and (not f_ground):
        with_out+=1
    if f_label and f_ground:
        both_have+=1 
print("在伪标签中有{},在groundtruth中有{},两个都没有{},两个都有：{}".format(in_pseudolabel,in_gt,with_out,both_have))
