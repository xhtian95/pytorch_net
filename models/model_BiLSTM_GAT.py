import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ptGAT_model import GAT, DNN
from BiLSTM import BLSTM

from torchvision import models

# 输入维度（batch，num_nodes节点数目，nfeat输入特征数）
# 输出维度（batch， num_nodes节点数目)
class BiLSTM_GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, in_drop, coef_drop, alpha, nheads, batch,
                 num_lstm, lstm_hidsize, lstm_numlayers):
        super(BiLSTM_GAT, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.alpha = alpha
        self.nheads = nheads
        self.num_lstm = num_lstm
        self.lstm_hidsize = lstm_hidsize
        self.lstm_numlayers = lstm_numlayers  # 循环网络的LSTM层数
        self.batch = batch
        # self.biFlag = biFlag

        for _ in range(num_lstm):
            self.bilstm = nn.LSTM(input_size=nfeat, hidden_size=lstm_hidsize, num_layers=lstm_numlayers,
                                   batch_first=True, bidirectional=True)

        # DNN: 输入维数nfeat, 输出维数nclass
        self.dnn1 = DNN(nfeat=lstm_hidsize*2, nhid=nhid, nclass=num_nodes)
        self.dropout1 = nn.Dropout(in_drop)

        self.gat1 = GAT(nfeat, nhid, nhid, in_drop, coef_drop, alpha, nheads, last_layer=False)
        self.dropout2 = nn.Dropout(in_drop)
        self.gat2 = GAT(nhid, nhid, nclass, in_drop, coef_drop, alpha, nheads)

    def forward(self, x):
        c0 = torch.rand(num_lstm*2, batch, lstm_hidsize)
        h0 = torch.rand(num_lstm*2, batch, lstm_hidsize)

        lstm_output, (h_out, c_out) = self.bilstm(x, (h0, c0))
        print(lstm_output.shape)
        # lstm_out输出维数为(batch, num_nodes, hidden_size*num_directions)

        adj_pred = self.dnn1(lstm_output)
        print("adj_pred")
        print(adj_pred.shape)
        drop_adj = self.dropout1(x)
        print("drop_adj")
        print(drop_adj.shape)
        # drop_adj直接使用bilstm+dropout的结果矩阵
        # ajd_pred直接使用的输入x的变换
        attenout1 = self.gat1(drop_adj, adj_pred)
        print("attenout1")
        print(attenout1.shape)
        dropoutput = self.dropout2(attenout1)

        outputs = self.gat2(dropoutput, adj_pred)
        print("outputs")
        print(outputs.shape)
        outputs = F.softmax(outputs, dim=1)

        return outputs


if __name__=="__main__":
    num_nodes = 30
    n_features = 40
    hidden_units = 8
    num_lstm = 2
    lstm_hidsize = 8
    batch = 3
    a = torch.rand(batch, num_nodes, n_features)
    print(a.size())
    print(a)
    net = BiLSTM_GAT(nfeat=n_features, nhid=hidden_units, nclass=1, in_drop=0, coef_drop=0, alpha=0.2, nheads=8, batch=3,
                 num_lstm=num_lstm, lstm_hidsize=8, lstm_numlayers=2)

    outputs = net(a)
    print(outputs)
    print(outputs.shape)

