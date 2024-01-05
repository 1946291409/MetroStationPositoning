'''
Descripttion: 查验新的数据 和 raw数据 是否一致，看看我们是否在处理过程中搞错了。解决一下GNN模型准确率大幅下降的问题
准备使用map的方式，key：距离信息  value：伪标签， 如果k-v对应一致的话，则数据没错。这里key我们只取前10维
Author: Xue
Date: 2023-11-19 20:14:54
LastEditTime: 2023-11-20 15:23:49
'''

# TODO 加载raw数据
def get_raw_data():
    raw_dict = dict()
    path = "/data/XueMengYang/MSP_new/XueProject/data/beijing-原始的数据/data.txt"
    with open(path ,'r') as f:
        line = f.readline().strip()
        while line :
            data = line.split('	')
            key_data =[str(int(float(d))) for d in data[1:31]]
            key = "-".join(key_data)
            value = data[-2]
            raw_dict[key] = value
            line = f.readline().strip()
    # print(raw_dict)        
    return raw_dict


# TODO 加载新版数据（伪标签数据即可）
def get_new_data():
    new_dict = dict()
    path = "/data/XueMengYang/MSP_new/XueProject/dataset/data/pseudo_label.txt"
    with open(path,'r') as f:
        line = f.readline().strip()
        while line :
            data = line.split(',')
            key_data =[str(int(float(d)) )for d in data[1:31]]
            key = "-".join(key_data)
            value = data[-2]
            new_dict[key] = value
            line = f.readline().strip()
    # print(new_dict)        
    return new_dict

    
# TODO 加载 地铁站原id 和新id 的映射关系数据
def get_mapping_data():
    path = "/data/XueMengYang/MSP_new/XueProject/data/new_data/subway_seq.txt"
    subway_mapping = dict()
    index = 0
    with open(path, 'r') as f:
        line = f.readline().strip()
        while line :
            subway_mapping[line] = index
            index+=1
            line = f.readline().strip()
    # print(subway_mapping)
    # print(len(subway_mapping))
    # print(index)
    return subway_mapping   # {旧：新}

def compare(raw_dict,new_dict,mapping:dict):
    mapping['-1'] = '-1'
    right_count = 0
    not_in_count = 0
    for rk,rv in raw_dict.items():
        if rv not in mapping or rv=='-1':
            not_in_count+=1
            continue
        rv = mapping[rv]
        if rk in new_dict and rv == new_dict[rk]:
            right_count += 1
    print(right_count,not_in_count,len(raw_dict),len(new_dict))
    
def main():
    raw_dict = get_raw_data()
    new_dict = get_new_data()
    mapping = get_mapping_data()
    compare(raw_dict,new_dict,mapping)
    


if __name__ == '__main__':
    main()