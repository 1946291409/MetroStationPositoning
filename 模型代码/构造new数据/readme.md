<!--
 * @Descripttion: 
 * @Author: Xue
 * @Date: 2023-11-20 21:16:31
 * @LastEditTime: 2023-11-22 14:15:53
-->

# 构造new数据
## 步骤
* 在原始数据（raw）中提取出我们新的（new）的数据集
* 对新数据集进行重映射

## 数据详情
* 基站图 gb_edges(weight)_data.txt
* 地铁站图 gs_edges_data.txt
* groundtruth gt_data.txt
* psuedo-label 伪标签数据 pse_data.txt

## output
/data/XueMengYang/MSP_new/XueProject/newdata

# 构造mask 数据，用于模型训练时 切分训练集、测试集等
* 代码位置:/data/XueMengYang/MSP_new/XueProject/code/构造new数据/mask数据
* 存储位置：/data/XueMengYang/MSP_new/XueProject/newdata/mask

# 构造X 、Y、edges、weights等特征数据、标签数据...
* 代码位置： /data/XueMengYang/MSP_new/XueProject/code/构造new数据/modeltrain数据
* 存储位置：/data/XueMengYang/MSP_new/XueProject/newdata/new/modeltrain_data

# 重映射数据
* 对基站、地铁站 id进行重映射
* 代码位置：/data/XueMengYang/MSP_new/XueProject/code/构造new数据/new数据-重映射
* 存储位置：/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex