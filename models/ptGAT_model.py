import torch
import torch.nn as nn
import torch.nn.functional as F

from gatattentionhead import GraphAttentionHead2


class DNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(DNN, self).__init__()
        self.W1 = nn.Linear(nfeat, nhid)
        self.W2 = nn.Linear(nhid, nhid)
        self.out = nn.Linear(nhid, nclass)

    def forward(self, x):
        num_nodes = x.shape[1]
        x = F.relu(self.W1(x))
        x = F.relu(self.W2(x))
        x = self.out(x)
        # x = x.reshape(-1, num_nodes)
        return x  # 与CrossEntropyLoss()对应


# 输入为矩阵a:(batch, nodes节点数，nfeat输入特征数)
# 邻接矩阵adj:(batch, nodes节点数，nodes）
# 输出为(batch, nodes节点数)
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, in_drop, coef_drop, alpha, nheads, last_layer=True):
        """dense version of GAT"""
        super(GAT, self).__init__()

        self.last_layer = last_layer

        self.first_attentions = [GraphAttentionHead2(nfeat, nhid, in_drop, coef_drop, alpha, concat=True)
                                for _ in range(nheads)]
        for i, attention in enumerate(self.first_attentions):
            # 将一个child module添加到当前module,被添加的module可以通过name属性来获取
            # self.add_module(name, module)
            self.add_module('attention_{}'.format(i), attention)
            # self.attention_i = first_attentions(i)

        # 第二层只有一个head
        # nclass=1为了降低最后输出(batch, nodes节点数，nclass输出特征数)，去除最后一个维度，使最终结果与节点数相关
        self.out_att = GraphAttentionHead2(nhid * nheads, nclass, in_drop, coef_drop, alpha, concat=False)

    def forward(self, x, adj):
        num_nodes = x.shape[1]

        # 横向拼接几个head的结果
        x = torch.cat([atten(x, adj)for atten in self.first_attentions], dim=2)

        # final layer/prediction layer, using average, 推迟非线性变换（softmax，logistic sigmoid）
        x = self.out_att(x, adj)
        # 输出维度：(图数目，节点数目即num_nodes)

        if self.last_layer==False:
            return x
        else:
            x = x.reshape(-1, num_nodes)
            return x  # 与crossentropyloss()对应
            # 输出维度：(图数目，节点数)
            # return F.log_softmax(x, dim=1)  与NLL Loss对应


if __name__ == '__main__':
    num_nodes = 30
    n_features = 40
    hidden_units = 8
    a = torch.rand(3, num_nodes, n_features)
    adj = torch.rand(3, num_nodes, num_nodes)

    net = GAT(nfeat=n_features, nhid=hidden_units, nheads=8, nclass=1, in_drop=0, coef_drop=0, alpha=0.2)
    out_vec = net(a, adj)
    print(out_vec.shape)
    print(type(out_vec))
