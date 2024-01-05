'''
考虑加入社区发现算法
检测是否有 社区发现 的性质：即 社区中的节点紧密相连，不同社区间的节点稀疏相连
考虑检测，groundtruth中的地铁站的x 所匹配的N个基站是否是"紧密相连"的（比如k-hop可达）

直觉上，距离上肯定是紧密的，但是基站图中不一定在该N个节点的子区域是稠密的。
'''

import networkx as nx


root_path = "/data/XueMengYang/MSP_new/XueProject/dataset"

# k-hop 可达
k = 3
