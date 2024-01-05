'''
查看基站图权重分布，去除基站图的权重中较小边
'''
import matplotlib.pyplot as plt
import networkx as nx
path = "../data/graph.txt"

def load_data():
    with open(path, 'r') as f:
        data = f.readline().strip()
        edges = []
        weights = []
        while(data):
            data = data.split(" ")
            b1 = data[0]
            b2 = data[1]
            w = data[2]
            edges.append([b1,b2])
            weights.append(int(w))
            data = f.readline().strip()
    
    return edges,weights




def draw(weights):
    # 绘制直方图
    data = [0]*1114
    for i in weights:
        data[i]+=1
    index = [x for x in range(len(data))]
    
    data_index_sorted = sorted(index,key=lambda i:data[i],reverse=True)
    data_sorted = sorted(data,reverse=True)
    
    print([round(x/len(weights),3) for x in data_sorted[:10] ])
    print(data_index_sorted[:10])
    # plt.hist(weights, bins= 10, density=True, alpha=0.7, color='g')
    # plt.savefig("fenbu.png")


def delete_small_weight(edges,weights,delete_arr):
    print("删除：",delete_arr)
    g = nx.Graph()
    for edge in edges:
        g.add_node(edge[0])
        g.add_node(edge[1])
    for i in range(len(weights)):
        if weights[i] not in delete_arr:
            g.add_edge(*edges[i])
    connected_components = list(nx.connected_components(g))
    connected_components.sort(reverse=True,key= lambda x : len(x))
    print([len(x) for x in connected_components ][:10],"total components:",len(connected_components))
    print("总的节点数",g.number_of_nodes())
    x = 0
    x = 0
    for c in connected_components:
        if len(c) == 1:
            x += 1
    print("只有一个节点的连通分量数：",x)

if __name__ == "__main__":
    edges,weights = load_data()
    #draw(weights)
    delete_small_weight(edges,weights,[2])