'''
Descripttion: 查验数据
Author: Xue
Date: 2023-11-13 16:00:38
LastEditTime: 2023-11-13 16:42:46
'''

import numpy as np

a = np.load("/data/XueMengYang/MSP_new/XueProject/dataset/feature_data/X.npy")
print('ok')



# 创建一个 n*m 的二维数组
x = np.array([[1, 2, 3], [4, 5, 6]])

# 对数组 x 进行逐行归一化
row_norms = np.linalg.norm(x, axis=1)
print(row_norms)

# 对数组 x 进行逐行归一化,并将其归一化值作为权重应用于相应的行
x_normalized = x / row_norms.reshape(-1, 1)

print("原始数组 x:")
print(x)
print("归一化后的数组 x_normalized:")
print(x_normalized)


print("======================")
import numpy as np
 
x = np.array([[10,  10,   10],
              [ 1,   5,  3],
              [ 8,   7,  1]])
print(x.shape)
x_norm = x / x.max(axis=1)
print(x.max(axis=1).shape)
print(x_norm)