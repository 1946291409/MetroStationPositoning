'''
Descripttion: 
Author: Xue
Date: 2023-11-25 10:55:34
LastEditTime: 2023-11-25 11:01:22
'''
from collections import defaultdict
import numpy as np

path = "/data/XueMengYang/MSP_new/XueProject/数据查验/未匹配正确的点/output/res_right_dis.npy"
dis = np.load(path)
# print(max(dis))
# 得到每种 距离尺度 的个数 
map = defaultdict(int)
for d in dis:
    map[d] += 1
dis_res = tuple(map.items())
dis_res = sorted(dis_res, key=lambda x: x[1],reverse=True)  
for item in dis_res:
    print("{}的占比为：{}".format(item[0],round(item[1]/len(dis),4)))
print(len(dis))
# exit()
import matplotlib.pyplot as plt
# a = [2,3,3,3,2,1,5,6]
plt.hist(dis,bins=max(dis),rwidth=0.9)
# print(time.time())
plt.savefig("/data/XueMengYang/MSP_new/XueProject/数据查验/未匹配正确的点/output/" + "test_res_right_dis.png" )
