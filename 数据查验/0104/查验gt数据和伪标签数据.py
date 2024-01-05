'''
Description: 检查伪标签数据 在 gt数据 中的准确率
Author: Xue
Date: 2024-01-04 15:07:39
LastEditTime: 2024-01-04 16:47:28
'''
import numpy as np

gt_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/gt_reindex.txt"
pseudo_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/model_train_data/re_Y.npy"


def check():
    # 加载gt数据
    gt_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/gt_reindex.txt"
    with open(gt_path ,'r') as f:
        gt_datas = f.readlines()
        gt_datas = [data.strip() for data in gt_datas]
        gt_datas = [data.split(',') for data in gt_datas]
    # print(gt_datas)
    
    # 加载伪标签
    pseudo_data = np.load(pseudo_path)
    
    # 检验准确率
    count = 0
    for b_s  in gt_datas:
        b_s = [int(i) for i in b_s]
        b = b_s[0]
        s = b_s[1]
        if pseudo_data[b] == s:
            count+=1
            print(f"{b} {s} 正确")
    print(count,len(gt_datas))
    

check()