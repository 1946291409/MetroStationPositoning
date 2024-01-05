'''
Description: gt 和 伪标签数据差距太大，是不是reindex时出了问题
Author: Xue
Date: 2024-01-04 15:34:26
LastEditTime: 2024-01-04 17:04:48
'''
import numpy as np

# 基站、地铁站的reindx 的map
def get_map():
    s_path ="/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/subway_re_ids.txt"
    b_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/base_re_ids.txt"
    s_map = dict()  # {new:old}
    b_map = dict()
    re_b_map = dict()   #{old:new}
    with open(s_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            s_map[int(line[1])] = int(line[0])
    with open(b_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            b_map[int(line[1])] = int(line[0])
            re_b_map[int(line[0])] = int(line[1])
    return s_map, b_map,re_b_map

s_map, b_map,re_b_map = get_map()
s_map[-1] = -1

print('ok')
# 检验grpundtruth数据reindex
def check_gt_reindex():
    gt_new_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/gt_reindex.txt" # b-s
    gt_old_path = "/data/XueMengYang/MSP_new/XueProject/data/beijing-原始的数据/bj_collect.txt" # b-s

    new_b_s = []
    with open(gt_new_path,'r') as f:   
        for line in f.readlines():
            line = line.strip().split(',')
            new_b_s.append([int(line[0]),int(line[1])])
    
    old_b_s = dict()
    with open(gt_old_path,'r') as f:   
        for line in f.readlines():
            line = line.strip().split(',')
            old_b_s[int(line[0])] = int(line[1])

    count = 0
    for i in range(len(new_b_s)):
        b = b_map[new_b_s[i][0]]
        s = s_map[new_b_s[i][1]]
        if b in old_b_s and old_b_s[b] == s:
            count+=1
    print(count,count/len(new_b_s))
    
# check_gt_reindex()
    


# 检验伪标签数据reindex
def check_pseudo_reindex():
    pseudo_new_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/model_train_data/re_Y.npy"
    pseudo_id_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new/model_train_data/id.npy"
    pseudo_old_path = "/data/XueMengYang/MSP_new/XueProject/data/beijing-原始的数据/data.txt"
    
    Y = np.load(pseudo_new_path)
    ids = np.load(pseudo_id_path) # 旧id
    # ids = [int(i) for i in ids] 
    new_b_s = np.vstack((ids,Y)).T
    # print(len(new_b_s))
    
    old_b_s = dict()
    with open(pseudo_old_path,'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            old_b_s[int(line[0])] = int(line[-2])
    # print(old_b_s)
    # print(len(old_b_s))
    
    count = 0
    for i in range(len(new_b_s)):
        b = int(new_b_s[i][0])
        s = s_map[int(new_b_s[i][1])]
        if b in old_b_s and old_b_s[b] == s:
            count+=1
    print(count,count/len(new_b_s)) 
    
    
# check_pseudo_reindex()
def reindex_ids():
    pseudo_id_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new/model_train_data/id.npy"
    ids = np.load(pseudo_id_path) # 旧id
    new_ids = [re_b_map[int(id)] for id in ids]
    ids = np.array(new_ids)
    np.save("/data/XueMengYang/MSP_new/XueProject/newdata/new/model_train_data/re_id",ids)
    print('ok')
    
reindex_ids()