'''
Descripttion: sklearn 快捷实现k-means算法， 准确率通过 簇内大部分点的归属判断   => 0.9100941356061569
Author: Xue
Date: 2023-11-14 15:46:12
LastEditTime: 2023-11-28 21:38:07
'''

from uu import Error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from collections import defaultdict

from zmq import Errno
# np.set_printoptions(threshold=1e6)

def vote(arr,y):
    vote_dict = defaultdict(lambda : 0)
    for a in arr:
        print(y[a])
        vote_dict[y[a]] += 1
    label = max(vote_dict,key= lambda k:vote_dict[k])
    right = 0
    for a in arr:
        if y[a] == label:
            right += 1
    return right

def correct(y_pred,y):
    cluster_dict = dict()
    # 初始化
    for i in range(337):
        cluster_dict[i] = []
        
    right_sum = 0   
    for i in range(len(y_pred)):
        cluster_dict[y_pred[i]].append(i)
    for _,arr in cluster_dict.items():
        right_sum += vote(arr,y)
    count = 0
    for i in y:
        if i!=-1:
            count += 1
    return right_sum/count



def main():
    n_samples = 337
    x = np.load("/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/model_train_data/X.npy")
    y = np.load("/data/XueMengYang/MSP_new/XueProject/code/GNN_baselines/output/res.npy")
    print(y)
    y = y[0]
    # exit()
    # x, y = make_blobs(n_samples=n_samples)
    #y_pred = KMeans(n_clusters=n_samples,init="k-means++").fit_predict(x)
    y_pred = np.load("/data/XueMengYang/MSP_new/XueProject/code/聚类/output/y_pred.npy")
    # np.save("/data/XueMengYang/MSP_new/XueProject/code/聚类/output/y_pred.npy",y_pred)
    print(y_pred)
    print(len(y),len(y_pred))
    acc = correct(y_pred,y)
    print("准确率为:",acc)
    
    
if __name__ == "__main__":  
    main()