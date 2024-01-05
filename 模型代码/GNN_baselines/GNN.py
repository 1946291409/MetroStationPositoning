'''
Descripttion: 
Author: Xue
Date: 2023-10-25 16:30:08
LastEditTime: 2024-01-04 17:13:39
'''
# 实现GCN 导入的包
from os import path
from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, ChebConv, GINConv, MLP, GINEConv, GIN, GCN2Conv
import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import time

# GCN: 线性层+GCN
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, bias=False):
        super(GCN, self).__init__()
        self.depth = 2
        self.convs = nn.ModuleList()
        feats = [in_feats,in_feats*2, out_feats]   # 两层GCN输入输出分别为   in->2*in 、 2*in->out
        for i in range(self.depth):
            self.convs.append(GCNConv(feats[i],feats[i+1]))
        # self.lstconv = GCNConv(in_dim, out_dim)
        
        self.fc = nn.Linear(feats[self.depth], out_feats)    # 全连接层,在GCNs 最后
        self.droupout_p = 0.3
        

    def forward(self, x, edge_index, edge_weight):
        for conv in self.convs:
            x = conv(x,edge_index,edge_weight)
            x = F.relu(x)
            x = F.dropout(x,p=self.droupout_p,training=self.training)
        # x = self.lstconv(x,edge_index,edge_weight)
        x = self.fc(x)
        return x

# mGCN



# GCNII:
# 定义GCNII_Net类，继承nn.Module
class GCNII_Net(nn.Module):
    # 初始化函数，设置参数
    def __init__(self,in_feats,out_feats, a=0.5):
        super(GCNII_Net, self).__init__()
        #if alpha == None:
        alpha = a
        # 初始化GCNIIs列表
        self.GCNIIs = nn.ModuleList()
        # 设置深度
        self.depth = 2
        # 遍历深度，添加GCN2Conv层
        for i in range(self.depth):
            self.GCNIIs.append(GCN2Conv(in_feats, alpha=alpha))
        # 初始化全连接层
        self.fc = nn.Linear(in_feats, out_feats)    # 全连接层

        # 设置dropout比例
        self.dropout = 0.3


    # 定义前向传播函数
    def forward(self, x, edge_index, edge_weight):
        # 复制输入x
        x_0 = x.clone()
        # 如果edge_weight不为空，添加维度
        # if edge_weight != None:
        #     edge_weight = edge_weight.unsqueeze(-1)
        # 遍历GCNIIs，进行前向传播
        for conv in self.GCNIIs:
            x = F.relu(conv(x, x_0, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
        # 进行全连接层
        x = self.fc(x)
        # 进行dropout
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # 进行卷积层
        # x = self.conv2(x, edge_index, edge_weight)
        # 返回结果
        return x



class GAT_Net(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GAT_Net, self).__init__()
        self.emb_dim = in_feats
        self.out_dim = out_feats
        self.num_heads = 2  # 多头 
        self.dropout = 0.3
        self.depth = 2
        self.convs = nn.ModuleList()
        feats = [in_feats,self.num_heads*in_feats,out_feats]
        
        self.conv1 = GATConv(
            in_feats,
            in_feats,
            heads=self.num_heads,
            edge_dim=1,
            dropout=self.dropout)
        self.conv2 = GATConv(
            in_feats * self.num_heads,
            out_feats,
            heads=self.num_heads,
            edge_dim=1,
            concat=False,
            dropout=self.dropout)
        
        
    # def reset_parameters(self):
    #     print('reset parameters')
    #     self.conv1.reset_parameters()
    #     self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        if edge_weight != None:
            edge_weight = edge_weight.unsqueeze(-1) # （N,） -> （N,1）
        # x = F.dropout(x, p=self.dropout, training=self.training)   
        x = F.relu(self.conv1(x, edge_index, edge_weight)) 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x



class GAT_Net_test(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GAT_Net_test, self).__init__()
        # 设置网络参数
        self.emb_dim = in_feats
        self.out_dim = out_feats
        self.num_heads = 2  # 多头 
        self.dropout = 0.3
        self.depth = 1
        self.convs = nn.ModuleList()
        ins = [in_feats, in_feats*self.num_heads]
        outs = [in_feats,in_feats]  # 每层实际的输出是 outs[i]*heads
        
        # 定义网络结构 2*GAT + 1*line 
        for i in range(self.depth):
            if i == self.depth - 1: # 最后一层GAT ，则不进行concate拼接，而是选择mean聚合。
                self.convs.append(GATConv(
                ins[i],
                outs[i],
                heads=self.num_heads,
                edge_dim=1,
                dropout=self.dropout,
                concat=False
                ))
            else :
                self.convs.append(GATConv(
                    ins[i],
                    outs[i],
                    heads=self.num_heads,
                    edge_dim=1,
                    dropout=self.dropout
                ))
        self.fc = nn.Linear(outs[self.depth-1]*self.num_heads, out_feats)
        
        
    def forward(self, x, edge_index, edge_weight):
        if edge_weight != None:
            edge_weight = edge_weight.unsqueeze(-1) # （N,） -> （N,1）
        # # x = F.dropout(x, p=self.dropout, training=self.training)   
        # x = F.relu(self.conv1(x, edge_index, edge_weight)) 
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.conv2(x, edge_index, edge_weight)
        for i,conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i == len(self.convs)-1:
                break
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.fc(x)
        return x




    
# 定义GIN_Net类，继承nn.Module
class GIN_Net(nn.Module):
    # 初始化函数，参数args为字典类型
    def __init__(self, in_feats, out_feats):
        super().__init__()
        # 获取深度
        self.depth = 2
        self.gins = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        mlp_layers = 2

        # 遍历深度
        for _ in range(self.depth - 1):
            # 初始化MLP
            mlp = torch.nn.Sequential(MLP(in_channels=in_feats,
                      hidden_channels=2 * in_feats,
                      out_channels=in_feats,
                      num_layers=mlp_layers))
            self.gins.append(GINEConv(nn=mlp, train_eps=True, edge_dim=1))

            self.batch_norms.append(nn.BatchNorm1d(in_feats))

        lstmlp = torch.nn.Sequential(MLP(in_channels=in_feats,
                      hidden_channels=2 * in_feats,
                      out_channels=out_feats,
                      num_layers=mlp_layers))
        # 添加GIN层
        self.gins.append(GINEConv(nn=lstmlp, train_eps=True, edge_dim=1))
        # 获取dropout
        self.dropout = 0.3

    # 定义前向传播函数
    def forward(self, x, edge_index, edge_weight):
        
        # 归一化、加自环
        edge_index, edge_weight = gcn_norm(
            edge_index, edge_weight, num_nodes=x.shape[0], add_self_loops=True)
        edge_weight = edge_weight.unsqueeze(-1)
        layer_outputs = []
        for i in range(self.depth - 1):
            x = self.gins[i](x, edge_index, edge_weight)
            x = F.relu(self.batch_norms[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)
        # 添加最后一个GIN层
        x = self.gins[-1](x, edge_index, edge_weight)
        # x = torch.stack(layer_outputs, dim=0)
        # x = torch.max(x, dim=0)[0]
        return x


# GPRGNN



def check_gt():
    gt_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/gt_reindex.txt"
    with open(gt_path ,'r') as f:
        gt_datas = f.readlines()



def RunExp():
    input_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/model_train_data/"
    X = np.load(input_path + "X_col_norm.npy")
    edges = np.load(input_path + "re_edges.npy")
    weights = np.load(input_path + "weights.npy",allow_pickle=True)
    # 加载基站的 输入数据
    gb_X = torch.from_numpy(X).float().to(device)   # 仅337距离
    quadkeys = np.load(input_path + "quadkeys.npy")
    gb_quadkeys = torch.from_numpy(quadkeys).float().to(device)
    gb_X = torch.cat([gb_X,gb_quadkeys],dim=1)    # 337距离 + quadkey
    # gb_X = gb_quadkeys  # 仅quadkey
    gb_edges = torch.from_numpy(edges).to(device)
    gb_weights = torch.from_numpy(weights).float().to(device)
    
    # 加载 训练、验证、测试 的掩码数据
    root_mask = "/data/XueMengYang/MSP_new/XueProject/newdata/mask/pse/"
    train_mask = np.load(root_mask + "train_mask.npy")
    valid_mask = np.load(root_mask + "valid_mask.npy")
    test_mask = np.load(root_mask + "test_mask.npy")
    train_mask = torch.from_numpy(train_mask).to(device)
    valid_mask = torch.from_numpy(valid_mask).to(device)
    test_mask = torch.from_numpy(test_mask).to(device)
    
    # 加载监督信息 y 的数据
    y = np.load(input_path + "re_Y.npy")
    y = torch.from_numpy(y).to(device)
    
    # 初始化模型、优化器、损失函数
    if model_name == "GCN":
        model = GCN(gb_X.size(1),337).to(device)
    elif model_name == "GCNII":
        model = GCNII_Net(gb_X.size(1),337).to(device)
    elif model_name == "GAT":
        model = GAT_Net_test(gb_X.size(1),337).to(device)
    elif model_name == "GIN":
        model = GIN_Net(gb_X.size(1),337).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
    
    if loss_name == "CE":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_name == "NLL":
        criterion = F.nll_loss
    
    
    
    # 训练
    def train(model, optimizer,criterion,x,edges,weights):
        model.train()
        optimizer.zero_grad()
        out = model(x,edges,weights)
        loss = criterion(out[train_mask],y[train_mask])
        # loss  = F.nll_loss(out[train_mask],y[train_mask])
        loss.backward()
        optimizer.step()
        return loss
    # 验证
    def valid(model,x,edges,weights):
        model.eval()
        out = model(x,edges,weights)
        pred = out.argmax(dim=1)
        corret = pred[valid_mask] == y[valid_mask] 
        acc = int(corret.sum()) / int(valid_mask.sum())
        # 打印测试结果
        print(f'Valid Accuracy: {acc:.4f}')
    # 测试
    def test(model,x,edges,weights):
        model.eval()
        out = model(x,edges,weights)
        pred = out.argmax(dim=1)
        corret = pred[test_mask] == y[test_mask] 
        error = pred[test_mask] != y[test_mask]
        # test 准确率
        acc = int(corret.sum()) / int(test_mask.sum())
        # 打印测试结果
        print(f'Test Accuracy: {acc:.4f}')
        
        # 保存test中所有node的结果 （节点id,预测结果,gt,T/F）
        if is_save :
            cur_time = time.strftime('%m月%d日%M:%S', time.localtime())
            res_save_path = "/data/XueMengYang/MSP_new/XueProject/模型代码/GNN_baselines/output/{}_{}_{}.npy".format(model_name,cur_time,round(acc,2))
            x = np.where(test_mask.cpu().numpy()==True)
            res = pred[test_mask].cpu().numpy()
            right = y[test_mask].cpu().numpy()
            res_all = np.vstack((x,res,right))
            np.save(res_save_path,res_all.T)
            print("结果保存成功")    
        
        
    # 在groundtruth 数据上检验准确率
    def check_gt(model,x,edges,weights):
        # 加载gt数据
        gt_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new_reindex/gt_reindex.txt"
        pseudo_id_path = "/data/XueMengYang/MSP_new/XueProject/newdata/new/model_train_data/re_id.npy"
        with open(gt_path ,'r') as f:
            gt_datas = f.readlines()
            gt_datas = [data.strip() for data in gt_datas]
            gt_datas = [data.split(',') for data in gt_datas]
        
        # 加载模型
        model.eval()
        out = model(x,edges,weights)
        pred = out.argmax(dim=1)
        
        ids = np.load(pseudo_id_path) # 旧id
        pred_dict = dict()
        for i,id in enumerate(ids):
            pred_dict[id] = pred[i]
        
        count = 0
        for b_s in gt_datas:
            b = int(b_s[0])
            s = int(b_s[1])
            if pred_dict[b] == s:
                count+=1
        print("在gt上的准确率为:{}".format(round(count/len(gt_datas),4)))

    
    # 进行训练
    for epoch in range(num_epochs):
        loss = train(model,optimizer,criterion,gb_X,gb_edges,gb_weights)
        # print("Epoch: {}, Loss: {:.2f}".format(epoch,loss))
        if epoch%20 == 0:
            print("Epoch: {}, Loss: {:.2f}".format(epoch,loss))
        if epoch%200 == 0:
            valid(model,gb_X,gb_edges,gb_weights)
        
    
    test(model,gb_X,gb_edges,gb_weights) 
    check_gt(model,gb_X,gb_edges,gb_weights)   
    
num_epochs = 1000
is_save = False
device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")
model_name = "GCN"  # GCN、GCNII 、GAT 、 GIN
loss_name = "CE"    # CE、NLL、QL、FL
def main():
    RunExp()


if __name__ == '__main__':
    main()

