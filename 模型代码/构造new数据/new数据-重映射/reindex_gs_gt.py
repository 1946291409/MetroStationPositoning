'''
Descripttion: 对gs和gt的原始edges 数据进行重映射
Author: Xue
Date: 2023-11-22 22:07:31
LastEditTime: 2023-11-22 22:08:38
'''


def get_reindex_gs():
    gs_map = dict()
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/subway_re_ids.txt"
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        data = line.strip().split(",")
        gs_map[data[0]] = int(data[1])
    return gs_map
    
def get_reindex_gb():
    gt_map = dict()
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/base_re_ids.txt"
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        data = line.strip().split(",")
        gt_map[data[0]] = int(data[1])
    # print(len(gt_map))
    return gt_map 

