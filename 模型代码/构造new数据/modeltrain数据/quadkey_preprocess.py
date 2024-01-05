'''
Descripttion: 将伪标签中的quadkey加入到特征中，即将其split为19维的特征数据（quadkey本来是长度为19的数字）
Author: Xue
Date: 2023-11-22 19:31:26
LastEditTime: 2023-11-22 19:32:34
'''


import mercantile
import numpy as np

def quadkey_to_ul(quadkey):
    # quadkey -> 瓦片位置
    tile = mercantile.quadkey_to_tile(quadkey)
    # 瓦片 -> 经纬度    
    return mercantile.ul(tile)  

def lat_lng_to_quadkey(lat, lng, zoom=19):
    atile = mercantile.tile(lat=lat, lng=lng, zoom=19)
    aqu = mercantile.quadkey(atile)
    return aqu

quadkeys = []
with open('/data/GeQian/MetroStationPositioning_2023/newdata/new/pse_data.txt', 'r') as f:
    lines = f.readlines()

X = np.load('/data/GeQian/MetroStationPositioning_2023/newdata/new_reindex/model_train_data/X.npy')

# print(quadkeys)


for line in lines:
    quad = line.strip().split(',')[-1]
    quad = [int(i) for i in quad]
    quadkeys.append(quad)

quadkeys = np.array(quadkeys)
print(quadkeys.shape)
np.save('/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/model_train_data', quadkeys)