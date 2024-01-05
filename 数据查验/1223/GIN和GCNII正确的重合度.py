'''
Description: GIN 和GCNII 模型的准确率都很高，其匹配准确的点有多少重合的。
Author: Xue
Date: 2023-12-23 20:59:51
LastEditTime: 2023-12-23 22:33:55
'''
import numpy as np


path_GCNII = '/data/XueMengYang/MSP_new/XueProject/code/GNN_baselines/output/GCNII_12月23日15:39_0.9.npy'
path_GIN = "/data/XueMengYang/MSP_new/XueProject/code/GNN_baselines/output/GIN_12月23日05:33_0.91.npy"

# 加载数据
def load_data():
    gcnii_data = np.load(path_GCNII)
    gin_data = np.load(path_GIN)
    return gcnii_data, gin_data 

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
    print("重合度为：{}".format(count/len(x1)))
    
gcnii_data, gin_data  = load_data()
x1 = get_right(gcnii_data)
x2 = get_right(gin_data)
coin(x1,x2)

