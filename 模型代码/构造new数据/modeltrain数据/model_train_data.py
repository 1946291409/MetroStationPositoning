'''
Description: 生成可以用于模型训练的 特征数据X、标签数据Y、边、权重数据
Author: Xue
Date: 2023-11-21 22:06:30
LastEditTime: 2023-11-22 11:18:31
'''


import numpy as np
save_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new/modeltrain_data/"

def get_id_X__Y():
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new/pse_data.txt"
    id = []
    X = []
    Y = []
    with open(path, "r") as f:
        line = f.readline().strip()
        while line:
            data = line.split(",")
            id.append(data[0])
            feature = [float(dis) for dis in data[1:-2]]
            X.append(feature)
            Y.append(data[-2])
            line = f.readline().strip()
    id = np.array(id)
    X = np.array(X)
    Y = np.array(Y)
    print(X)
    print(Y)
    np.save(save_path + "id.npy", id)
    np.save(save_path + "X.npy", X)
    np.save(save_path + "Y.npy", Y)
def get_X_norm():
    path = save_path + "X.npy"
    X = np.load(path)
    X_col_norm = X / X.max(axis=0)
    np.save(save_path + "X_col_norm.npy", X_col_norm) 
    print("okk")
get_X_norm()   

# get_id_X__Y()

# get_X_feature__Y()
def get_edges_weights():
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new/gb_edges(weight)_data.txt"
    edges = [[],[]]
    weights = []
    with open(path, "r") as f:
        line = f.readline().strip()
        while line:
            data = line.split(",")
            edges[0].append(data[0])
            edges[1].append(data[1])
            weights.append(float(data[2]))
            line = f.readline().strip()
    #print(edges)
    #print(weights)
    print(len(edges[0]) == len(weights))
    np.save(save_path + "edges.npy", edges)
    np.save(save_path + "weights.npy", weights)
    
# get_edges_weights()
    

