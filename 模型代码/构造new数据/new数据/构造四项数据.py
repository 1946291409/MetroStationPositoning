'''
Descripttion: 构造四项new数据：地铁站图、基站图、groundtruth、伪标签数据
Author: Xue
Date: 2023-11-21 15:56:51
LastEditTime: 2023-11-21 21:30:31
'''



from collections import defaultdict
import numpy as np
import os 
import pickle

root_path = "/data/XueMengYang/MSP_new/XueProject/data/beijing-原始的数据/"
save_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new/"    #格式为pickle


# 地铁站图节点
# seq（339维距离数据） 数据
def get_seq_data():
    path = root_path + 'station_seq.txt'
    seq_set = set()
    with open(path,'r') as f:
        line = f.readline().strip()
        while line:
            seq_set.add(line)
            line = f.readline().strip()
    # print(len(seq_set))
    return seq_set

def get_subway_data():
    path = root_path + "bj_station_trans_dict_2021.pkl"
    subway_set = set()
    subway_edge_set = set()
    seq_set = get_seq_data()
    with open(path ,'rb') as f:
        data = pickle.load(f)
    for key,values  in data.items():
        if key not in seq_set: continue
        else :
            subway_set.add(key)
            for v in values:
                if v not in seq_set or v == key: continue
                else :
                    subway_set.add(key)
                    subway_edge_set.add((min(key,v),max(key,v)))
    # print(len(subway_set))
    # print(len(subway_edge_set))
    return subway_set,subway_edge_set # gs_nodes,gs_edges
# get_subway_data()


def write_gs():
    save_gs = save_path + 'gs_edges_data.txt'
    _, edges_data = get_subway_data()
    with open(save_gs,'w') as f:
        for item in edges_data:
            f.write(item[0]+","+item[1] + '\n')
    print('gs save ok {}'.format(len(edges_data)))

# write_gs()

# 基站图节点
def write_gb():
    save_gb = save_path + 'gb_edges(weight)_data.txt'
    path = root_path + "graph.txt"
    gb_edge_dict = defaultdict(list)
    gb_nodes = set()
    with open(path ,'r') as f:
        line = f.readline().strip()
        while(line):
            data = line.split(" ")
            data = sorted(data[:2]) + [data[2]] # 对基站id排序
            gb_edge_dict[tuple(data[:2])].append(float(data[2]))
            gb_nodes.add(data[0])
            gb_nodes.add(data[1])
            line = f.readline().strip()
    
    # print(len(gb_data),len(gb_data_list),len(gb_edge_dict))
    
    # 对多个weight的情况 取均值
    for edge,w in gb_edge_dict.items():
        gb_edge_dict[edge] = sum(w)/len(w)
    print(len(gb_edge_dict),len(gb_nodes))

    # 保存gb_edges => 权重用float表示 
    if not os.path.exists(save_gb):
        with open(save_gb,'w') as f:
            for edge,w in gb_edge_dict.items():
                f.write(edge[0]+','+edge[1]+','+str(w)+'\n')
        print('gb save ok {}'.format(len(gb_edge_dict)))
        
    return gb_nodes

# write_gb()

# groundtruth
def write_gt_data():
    path = root_path + "bj_collect.txt"
    save_gt = save_path + 'gt_data.txt'
    gb_nodes = write_gb()
    gs_nodes,_ = get_subway_data()
    gt_list = []
    with open(path ,'r') as f:
        line = f.readline().strip()
        while(line):
            b,s= line.split(",")
            if b in gb_nodes and s in gs_nodes:
                gt_list.append([b,s])
            line = f.readline().strip()
    # print(len(gt_list))
    
    # 保存gt
    if not os.path.exists(save_gt):
        with open(save_gt,'w') as f:
            for t in gt_list:
                f.write(t[0]+','+t[1]+'\n')
        print('gt save ok {}'.format(len(gt_list)))
    
# write_gt_data()

# 伪标签数据
def write_pse_data():
    path = root_path + 'data.txt'
    save_pse = save_path + "pse_data.txt"
    gs_nodes,_ = get_subway_data()
    gb_nodes = write_gb()
    pse_list = []
    count_none  = 0
    with open(path, 'r') as f:
        line = f.readline().strip()
        while line:
            data = line.split('\t')
            b,s = data[0],data[-2]
            if b in gb_nodes:  
                data.pop(231)
                data.pop(230)
                if s not in gs_nodes:
                    s = '-1'
                    count_none+=1
                data[-2] = s
                pse_list.append(data)
                
            line = f.readline().strip()
    print(len(pse_list), "items in the list, with", count_none, "None values")   

    # 保存伪标签数据
    if not os.path.exists(save_pse):
        with open(save_pse,'w') as f:
            for data in pse_list:
                content = ""
                for item in data:
                    content = content + item + ","
                content = content[:-1]
                content += "\n"
                # print(content)
                f.write(content)
        print("save pse_data ok")   
    
    
# write_pse_data()


def check():
    gs_nodes,_ = get_subway_data()
    print('110000799560604' in gs_nodes)
    print('110000799560605' in gs_nodes)


# check()

def main():
    pass


if __name__ == "__main__":
    main()

