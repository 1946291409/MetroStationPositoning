'''
Description: 先使用GCNII进行训练，得到基站的预测标签。然后利用基站的距离信息进行聚类。聚类结果依据簇中点的投票结果来确定。
Author: Xue
Date: 2023-11-28 19:52:09
LastEditTime: 2023-11-30 16:58:38
'''

import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import os


# np.set_printoptions(threshold=1e6)



def vote(arr,y_gnn,y_right,y_change):
    # print(arr)
    vote_dict = defaultdict(lambda : 0)
    for a in arr:
        # print(y[a])
        vote_dict[y_gnn[a]] += 1
    if len(vote_dict) == 0:
        return 0
    # 投票结果
    label = max(vote_dict,key= lambda k:vote_dict[k])
    
    # 更改y的label -> y_change
    for a in arr:
        y_change[a] = label
    
    # 计算这个簇中正确的数量
    right = 0
    for a in arr:
        if y_right[a] == label:
            right += 1
    return right

def correct(y_pred,y_gnn,y_right,y_change):# 聚类结果、GNN结果、伪标签
    cluster_dict = dict()
    # 初始化
    for i in range(337):
        cluster_dict[i] = []
        
    right_sum = 0   
    for i in range(len(y_pred)):
        cluster_dict[y_pred[i]].append(i)
    for _,arr in cluster_dict.items():
        right_sum += vote(arr,y_gnn,y_right,y_change)
    
    # 统计 非-1 的总数
    count = 0
    for i in y_right:
        if i!=-1:
            count += 1
    print("gnn-km准确率为:",right_sum/count)
    
    right = 0
    count = 0
    for i in range(len(y_gnn)):
        if y_right[i] != -1:
            count += 1
            if y_right[i] == y_gnn[i]:
                right += 1
    
    print("gnn全体准确率为:", right/count)
    return right_sum/count

def km_mf(m=337):
    x_path = ""
    if m == 337:
        x_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/model_train_data/X.npy"
    elif m == 19:
        x_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/model_train_data/quadkeys.npy"
    elif m == 1:
        x_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/model_train_data/quadkeys_聚合.npy"
    else :
        print("特征维度错误，重新输入")
        return
    x = np.load(x_path)
    # 计算并保存结果
    y_pred = None
    save_path = "/data/XueMengYang/MSP_new/XueProject/code/聚类/output/km_{}.npy".format(m)
    if not os.path.exists(save_path):
        y_pred = KMeans(n_clusters=n_samples,init="k-means++").fit_predict(x)
        np.save(save_path,y_pred)
    else :
        y_pred = np.load(save_path)
    
    y_afterchange = [-1] * len(y_pred)
    
    # 计算准确率  
    print("特征维度为{}的结果为：".format(m))
    correct(y_pred,y_gnn,y_right,y_afterchange)
    
    # print(y_afterchange)
    change_save_path = "/data/XueMengYang/MSP_new/XueProject/code/聚类/output/y_change_{}.npy".format(m)
    np.save(change_save_path,y_afterchange)
    
    
    

y_gnn = None
y_right = None

n_samples = 337
def main():
    global y_gnn
    global y_right
    
    y_gnn = np.load("/data/XueMengYang/MSP_new/XueProject/code/GNN_baselines/output/res.npy")
    y_right = np.load("/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/model_train_data/re_Y.npy")
    
    mfs = [337,19,1]
    for m in mfs:
        # 加载结果
        km_mf(m)
        return
        

if __name__ == "__main__":  
    main()