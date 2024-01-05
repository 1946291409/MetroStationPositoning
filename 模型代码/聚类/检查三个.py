'''
Description: gnn结果，gnn+kmeans， 伪标签结果；分别检验gnn、gnn+km准确率；确定哪些点改错了，哪些改对了
Author: Xue
Date: 2023-11-30 14:33:36
LastEditTime: 2023-11-30 19:11:56
'''

from itertools import count
import numpy as np

gnn_path = "/data/XueMengYang/MSP_new/XueProject/code/GNN_baselines/output/res.npy"
gnn_km_path = "/data/XueMengYang/MSP_new/XueProject/code/聚类/output/y_change_337.npy"
pse_label_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/model_train_data/re_Y.npy"

def load_data():
    gnn = np.load(gnn_path)
    gnn_km = np.load(gnn_km_path)
    pse = np.load(pse_label_path)
    return gnn, gnn_km, pse
# 检查准确率
def check_accuracy(gnn, gnn_km, pse):
    gnn_correct = 0
    gnn_km_correct = 0
    count = 0
    for i in range(len(gnn)):
        if pse[i] == -1:
            continue
        count += 1
        if gnn[i] == pse[i]:
            gnn_correct += 1
        if gnn_km[i] == pse[i]:
            gnn_km_correct += 1
    print("gnn accuracy: ", gnn_correct / count)
    print("gnn+km accuracy: ", gnn_km_correct / count)

# 检查哪些基站点的结果改变了
def check_change(gnn, gnn_km, pse):
    count = 0
    change_count = 0
    for i in range(len(gnn)):
        if pse[i] == -1:
            continue
        count += 1
        if gnn[i] != gnn_km[i]:
            change_count += 1
    print("change count: ", change_count,change_count/count)

# 统计，“改变” 的结果。 对的改错的多少。错的改对的多少。错的改错的多少。
def check_change_count(gnn, gnn_km, pse):
    err_err = []
    err_rig = []
    rig_err = []
    for i in range(len(gnn)):
        if gnn[i] == gnn_km[i]:
            continue
        else :
            if gnn[i] != pse[i]:
                if gnn_km[i] == pse[i]:
                    err_rig.append(i)
                else:
                    err_err.append(i)
            else:
                if gnn_km[i] != pse[i]:
                    rig_err.append(i)
    print("err_err: ", len(err_err))
    print("err_rig: ", len(err_rig))
    print("rig_err: ", len(rig_err))




def main():
    gnn, gnn_km, pse = load_data()
    check_accuracy(gnn, gnn_km, pse)
    check_change(gnn, gnn_km, pse)
    check_change_count(gnn, gnn_km, pse)
if __name__ == "__main__":
    main()