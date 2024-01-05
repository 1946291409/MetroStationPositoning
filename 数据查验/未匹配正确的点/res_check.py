'''
Descripttion: 比较 GNN模型 结果中匹配错的点的 结果，即模型匹配的地铁站和实际匹配的地铁站之间的距离
Author: Xue
Date: 2023-11-24 20:15:57
LastEditTime: 2023-11-25 10:54:35
'''
from collections import defaultdict
import numpy as np
import networkx as nx 

def get_gs():
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/gs_edges_reindex.npy"
    data = np.load(path)
    # print(data)
    gs = nx.Graph()
    for i in range(337):
        gs.add_node(i)
    gs.add_edges_from(data) 
    print(gs)
    return gs

def draw():
    path = "/data/XueMengYang/MSP_new/XueProject/数据查验/未匹配正确的点/output/res_right_dis.npy"
    dis = np.load(path)
    print(max(dis))
    # return
    import matplotlib.pyplot as plt
    plt.hist(dis)
    # print(time.time())
    plt.savefig("/data/XueMengYang/MSP_new/XueProject/数据查验/未匹配正确的点/output/" + "res_right_dis.png" )



def get_distance():
    path = "/data/XueMengYang/MSP_new/XueProject/code/GNN_baselines/output/res.npy"
    res = np.load(path)

    # 加载gs图
    gs = get_gs()
    dis = []    # 距离
    no_path = []    # 存储不连接的情况
    for i in range(len(res[0])):
        if nx.has_path(gs, res[0][i], res[1][i]):
            d = nx.shortest_path_length(gs, source=res[0][i], target=res[1][i])
            dis.append(d)
        else :
            no_path.append([res[0][i],res[1][i]])
    print(no_path)
    path = "/data/XueMengYang/MSP_new/XueProject/数据查验/未匹配正确的点/output/res_right_dis.npy"
    # np.save(path,dis)
    
    # 得到每种 距离尺度 的个数 
    map = defaultdict(int)
    for d in dis:
        map[d] += 1
    dis_res = tuple(map.items())
    dis_res = sorted(dis_res, key=lambda x: x[1],reverse=True)  
    for item in dis_res:
        print("{}的占比为：{}".format(item[0],round(item[1]/len(dis),4)))
    print(len(dis))
    draw()