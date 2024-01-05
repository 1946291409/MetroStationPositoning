'''
Descripttion: 对伪标签数据构造 mask 数据，用于模型训练时切分 训练集、验证集、测试集
Author: Xue
Date: 2023-11-21 21:31:42
LastEditTime: 2023-11-22 20:17:29
'''
import numpy as np
import random


# 这里的data只是 伪标签信息不为 -1 的数据,来构成我们的监督信息（伪标签）
def split_data():
    train_ratio,valid_ration,test_ratio = 0.6,0.2,0.2
    # 加载伪标签  Y 数据
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new/modeltrain_data/Y.npy"
    Y = np.load(path)
    label_indices = [index for index, value in enumerate(Y) if value != '-1']   # 提取出其中 非-1 的伪标签的 index
    len_data = len(Y)   # 39490
    len_label = len(label_indices)   # 31444
    # print(label_indices)
    random.shuffle(label_indices)
    random_label_indices = label_indices    # 打乱重排
    # print(random_label_indices)
    # return
    # 训练集
    train_set = random_label_indices[:int(len_label*train_ratio)]
    # 测试集
    valid_set = random_label_indices[int(len_label*train_ratio):int(len_label*(train_ratio+valid_ration))]
    # 验证集
    test_set = random_label_indices[int(len_label*(train_ratio+valid_ration)):]
    
    train_set = get_array(train_set,len_data)
    valid_set = get_array(valid_set,len_data)
    test_set = get_array(test_set,len_data)
    
    save_path = "/data/XueMengYang/MSP_new/XueProject/newdata/mask/pse/"
    np.save(save_path+"train.npy",train_set)
    np.save(save_path+"valid.npy",valid_set)
    np.save(save_path+"test.npy",test_set)
    print("okkk")
    
    
def get_array(data,len_data):
    arr = np.zeros(len_data,dtype=bool)
    for a in data:
        arr[a] = True
    return arr
    
# split_data()
  
def check():
    train = np.load("/data/XueMengYang/MSP_new/XueProject/newdata/mask/pse/train.npy")
    print(train)  
    print(train.shape)
    print(np.count_nonzero(train))
# check()