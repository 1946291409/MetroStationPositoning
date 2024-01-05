'''
Description: GCN 和GAT 模型的准确率都很低，其匹配准确的点有多少重合的。以及 匹配低的模型 匹配上的点 是否是 匹配高的模型 的子集
Author: Xue
Date: 2024-01-03 21:27:08
LastEditTime: 2024-01-03 21:43:55
'''

import numpy as np


path_GCNII = '/data/XueMengYang/MSP_new/XueProject/模型代码/GNN_baselines/output/GCNII_12月23日15:39_0.9.npy'
path_GIN = "/data/XueMengYang/MSP_new/XueProject/模型代码/GNN_baselines/output/GIN_12月23日05:33_0.91.npy"
path_GCN = "/data/XueMengYang/MSP_new/XueProject/模型代码/GNN_baselines/output/GCN_01月03日26:48_0.17.npy"
path_GAT = "/data/XueMengYang/MSP_new/XueProject/模型代码/GNN_baselines/output/GAT_01月03日28:03_0.24.npy"


# 加载数据
def load_data():
    gcnii_data = np.load(path_GCNII)
    gin_data = np.load(path_GIN)
    gcn_data = np.load(path_GCN)
    gat_data = np.load(path_GAT)
    return gcnii_data, gin_data,gcn_data,gat_data

# 获取其中匹配正确的节点id
def get_right(data):
    right_x = []
    for arr in data:
        if arr[1] == arr[2]:
            right_x.append(arr[0])
    return right_x


# 获取匹配错误的节点id
def get_err(data):
    err_x = []
    for arr in data:
        if arr[1] != arr[2]:
            err_x.append(arr[0])
    return err_x


# 查看重合度
def coin(x1,x2):
    count = 0
    for i in x1:
        if i in x2:
            count += 1  
    return "重合度分别为：{} 和 {}".format(round(count/len(x1),2),round(count/len(x2),2))
    
gcnii_data, gin_data,gcn_data,gat_data  = load_data()
gcnii_r = get_right(gcnii_data)
gin_r = get_right(gin_data)
gcn_r = get_right(gcn_data)
gat_r = get_right(gat_data)

data = [gcnii_r,gin_r,gcn_r,gat_r]
data_name = ["GCNII","GIN","GCN","GAT"]
for i in range(len(data)):
    for j in range(i+1,len(data)):
        print(data_name[i],data_name[j],coin(data[i],data[j]))


