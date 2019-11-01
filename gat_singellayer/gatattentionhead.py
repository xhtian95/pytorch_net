import torch
import torch.nn as nn
import torch.nn.functional as F


# concat确定是前几层head直接并行处理出特征结果还是最后一层取平均数后再进行sigmoid取结果
class GraphAttentionHead2(nn.Module):
    def __init__(self, in_features, out_features, in_drop, coef_drop, alpha, concat=True):

        super(GraphAttentionHead2, self).__init__()
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features)  # (in_features, out_features)维度的权重矩阵
        self.a = nn.Linear(2 * out_features, 1)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):

        F.dropout(input, self.in_drop, training=self.training)  # 对输入的drop

        h = self.W(input)

        nb_graphs = h.size()[0]  # 图的数量
        N = h.size()[1]  # nodes num 节点数量

        # print(h.shape)
        a_input = torch.cat([h.repeat(1, 1, N).view(nb_graphs, N * N, -1),
                             h.repeat(1, N, 1)], dim=2).view(nb_graphs, N, N, 2 * self.out_features)
        # view（）相当于reshape()
        # squeeze(压缩维度))
        # print(a_input.shape)
        e = self.leakyrelu(self.a(a_input).squeeze(3))
        print(e.shape)
        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)  # 判断adj元素，大于0的attention设为e，否则设为0
        # softmax
        attention = F.softmax(attention, dim=2)
        # drop
        attention = F.dropout(attention, self.coef_drop, training=self.training)
        print(attention.shape  )
        h_prime = torch.matmul(attention, h)

        if self.concat:
            # apply a nonlinearity
            # 类似于RELU的激活函数
            return F.elu(h_prime)
        else:  # 如果是average,则推迟非线性变换
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == "__main__":
    nodes = 30
    in_features = 40
    out_features = 8
    a = torch.rand(3, nodes, in_features)
    adj = torch.rand(3, nodes, nodes)
    myNet = GraphAttentionHead2(in_features=in_features, out_features=out_features, in_drop=0,
                                coef_drop=0, alpha=0.2, concat=True)
    out_vec = myNet(a, adj)
    print(out_vec.shape)
