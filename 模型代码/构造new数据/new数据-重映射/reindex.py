'''
Descripttion: 对 基站id、地铁站id进行重映射
Author: Xue
Date: 2023-11-22 14:14:13
LastEditTime: 2023-11-22 14:31:34
'''
import os
# 加载基站 node 数据
def get_base_ids():
    # 获得ids
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new/gb_edges(weight)_data.txt"
    ids = set()
    with open(path, "r") as f:
        line = f.readline().strip()
        while line:
            data = line.split(",")
            ids.add(data[0])
            ids.add(data[1])
            line = f.readline().strip()
    # print(len(ids))
    
    # 重映射
    map = []
    index = 0
    for id in ids:
        map.append([id,index])
        index+=1
    # print(map)
    
    # 保存
    save_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/" + "base_re_ids.txt"
    if not os.path.exists(save_path):
        with open (save_path, "w") as f:
            for item in map:
                f.write(str(item[0])+","+str(item[1])+"\n")
        print("save ok")
get_base_ids()


# 加载地铁站 node 数据
def get_subway_ids():
    path = "/data/XueMengYang/MSP_new/XueProject/newdata/new/gs_edges_data.txt"
    ids = set()
    with open(path, "r") as f:
        line = f.readline().strip()
        while line:
            data = line.split(",")
            ids.add(data[0])
            ids.add(data[1])
            line = f.readline().strip()
    # print(len(ids))
    
    # 重映射
    map = []
    index = 0
    for id in ids:
        map.append([id,index])
        index+=1
    # print(map)
    
    save_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/" + "subway_re_ids.txt"
    if not os.path.exists(save_path):
        with open (save_path, "w") as f:
            for item in map:
                f.write(str(item[0])+","+str(item[1])+"\n")
        print("save ok")
    
    
get_subway_ids()