# 数据

## 数据说明

![image](https://github.com/1946291409/MetroStationPositoning/assets/57052375/f16b0090-6b45-4c4b-a3cf-567ad9e8673f)


（110000799560604、110000799560605 两个地铁站***不在gs中***，但在伪标签字段中）

* 右下角是说，在groundtruth中，有的基站匹配了多个地铁站。
* 伪标签数据：39490条（未删减）（每行长度340）

  * 基站id：1 维
  * 距离信息：337 维
  * 伪标签：1 维 （其中 8046个为-1，其余为地铁站id）
  * quadkey：1维

* groundtruth：1926 条(基站，地铁站)
* 基站图：带权，39490节点，205302条边（txt中有`258155`行，是因为有一些重复的edge）（存在 某一条边有多个不同权重的情况=> 权重取均值处理）

  * 基站id，基站id，权重（float）
  * 连通性：36000在一起，其他的都是（<32）的连通分量
* 地铁站图：337个节点，385条边


## 数据位置

* 对基站、地铁站重新映射index的数据：<kbd>/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex</kbd>

  * 其中，<kbd>model_train_data 文件夹</kbd> 是模型训练的数据，比如特征X、edges、weights等
  * 其余数据是：

    * <kbd>base_re_ids.txt</kbd>: 原始id 和新id 的映射关系
    * <kbd>gb_edges_reindex.npy</kbd>、<kbd>gt_reindex.txt</kbd>\:映射后的edges、groundtruth数据




# 代码

* 路径：<kbd>/data/XueMengYang/MSP_new/XueProject/模型代码/GNN_baselines</kbd>
* 目前的GNN-baselines(GCN、GAT、GCNII、GIN\)均在<kbd>GNN.py</kbd>中
