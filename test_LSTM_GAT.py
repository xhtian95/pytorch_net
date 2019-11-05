import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import pickle
import os
from model_BiLSTM_GAT import BiLSTM_GAT
from charlevel_BiLSTM_GAT import charlevel_BiLSTM_GAT
import datetime
import time
import math
from generate_input import util_tools
from net_input_tools import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # 不使用GPU


# 加载训练数据
train_dataset_path = "../LSTM_gat_data/labeled_june-train-sub5000.csv"


# 输出目录及文件名
def output_file(csv_path, new_word):
    out_file_name = os.path.split(csv_path)[1]
    out_file_name = out_file_name[:out_file_name.rfind('.')]  # 不包括后缀名
    out_file_name_new = out_file_name + new_word

    out_file_dir = '../'  # 文件相对位置
    print("out_name: ", out_file_name)
    print("out_name_new", out_file_name_new)
    # 输出文件的目录及文件名
    return out_file_dir, out_file_name_new


"""=================提取目前已知文件中所有的ID和alarm name================"""
dataset_path = "../LSTM_gat_data/labeled-June-10min(P)-RAN-20191023.csv"
df = pd.read_csv(dataset_path, header=0, index_col=None)

# 创建类别_lines字典，各个语言名字的列表
category_lines = {}
all_categories = []

df_1 = df.loc[:, ['ID', 'Alarm Name', 'Root Alarm']]
print("前十项: 'id，告警名称, 根告警'")
print(df_1[:10])
df_1['ID'] = df_1['ID'].astype(np.int)

id_input = df_1['ID'].values
name_input = df_1['Alarm Name'].values
root_input = df_1['Root Alarm'].values  # 类型numpy.ndarray
rows = df_1['ID'].unique()  # 去掉重复元素
"""全局的alarm name去重列表，可作为表头或alarm name字典"""
cols = df_1['Alarm Name'].unique()
cols = list(cols)
print(type(cols))

# 总文件中出现的所有ID和alarmname的个数（去重后）
all_num_id = len(rows)
all_num_alarm = len(cols)

# 输出表格所有列的数量
entire_num_list = len(id_input)


"""===========================将名字转化为tensor类型==============================="""
# ids为id，alarm name的字典，每一个id对应其序列中alarmname组成的列表
# id_input, name_input分别为为去重的id和alarm name 数组，两数组的类型numpy.ndarray
def generate_ids_listi(id_input, name_input):
    ids = {}
    k = 1
    listi = []
    for i in range(len(id_input)):
        if id_input[i] > k:
            ids[k] = listi
            listi = []
            k += 1
        if id_input[i] == k:
            listi.append(name_input[i])
        if i == len(id_input)-1:
            ids[k] = listi
    return ids

# 全局的{id: alarm name list}字典
ids = generate_ids_listi(id_input, name_input)

#print('ids:')
#str_ids = str(ids)
#print(str_ids)


# 确定alarm在名称列表dic中的编号
def alarmToIndex(alarmname, dic):
    """
    for i, x in enumerate(dic):
        if x == alarmname:
            index = i
    """
    index = dic.index(alarmname)
    return index


# 将alarm转化为<1 * n_letters>张量(one-hot?)
def alarmToTensor(alarmname, dic):
    tensor = torch.zeros(1, all_num_alarm)
    tensor[0][alarmToIndex(alarmname, dic)] = 1
    return tensor


"""idsToTensor为全部alarm的tensor"""
# 将line转换成<line_length * 1 * n_letters>张量或者one-hot字母向量的数列
# nameIndexDic为原始去重alarm name的列表，以此获得纵坐标（对应alarm name的索引）
def idsToTensor(iddics, nameIndexDic):
    tensor = torch.zeros(entire_num_list, 1, all_num_alarm)
    pddataframe = pd.DataFrame(np.zeros((entire_num_list, all_num_alarm), dtype=np.int), columns=cols)
    k = 0
    for allist in iddics.values():
        # print(allist)
        for a in allist:
            # print(a)
            # print(k)
            if k > 417193:
                print(a)
            tensor[k][0][alarmToIndex(a, nameIndexDic)] = 1
            # 将tensor写入excel，则需要取消中间维度，得到二维数据
            # pddataframe.at[k, a] = pddataframe.at[k, a] + 1
            k += 1

    return tensor, pddataframe


# 所有id alarmname 对应的tensor
tensor_ids, dataframe = idsToTensor(ids, cols)
"""
测试tensor的输出正确与否
for i in range(417180, 417201):
    print("k:", i)
    for a in cols:
        if tensor_ids[i][0][alarmToIndex(a, cols)] == 1:
            j = a
    print(j)
"""
# outfile_dir, outfile_name = output_file(dataset_path, '_tensor')
# print(tensor_ids.shape)
# dataframe.to_csv(os.path.join(outfile_dir, outfile_name + ".csv"), header=True, index=False)


# 将每个id出现的alarm列表转换成<list_length * 1 * n_letters>张量或者one-hot字母向量的数列
# idlist: 一个id的所有alarmname的列表，nameIndexDic为alarmname的索引字典
def siidToTensor(idlist, nameIndexDic):
    tensor = torch.zeros(len(idlist), 1, all_num_alarm)
    for li, alarm in enumerate(idlist):
        tensor[li][0][alarmToIndex(alarm, nameIndexDic)] = 1
    return tensor


"""===========================ID名称对应出现Alarm Name的计数==============================="""

IDSuccessive = False  # 输入文件中ID是否连续（训练数据ID很可能不连续）
binarizationFlag = True  # 输出的矩阵是否为二值


# ID名称对应出现Alarm Name的计数矩阵（01矩阵为二值矩阵）
def extractIDandAlarmname(dfile, IDSuccessive, binarizationFlag=True):
    df_1 = dfile.loc[:, ['ID', 'Alarm Name', 'Root Alarm']]
    # print("前十项: 'id，告警名称, 根告警'")
    # print(df_1[:10])
    df_1['ID'] = df_1['ID'].astype(np.int)

    if IDSuccessive:
        # alarm_list, id_list：去重的id和alarm name列表，类型numpy.ndarray
        alarm_list, id_list, net_input = generate_input3(df_1)
        _, _, net_input_01 = generate_input3(df_1, binarizationFlag=True)
    else:
        alarm_list, id_list, net_input = generate_input4(df_1)
        _, _, net_input_01 = generate_input4(df_1, binarizationFlag=True)

    return alarm_list, id_list, net_input, net_input_01


"""==================================training========================================"""
train_df = pd.read_csv(train_dataset_path, header=0, index_col=None)
train_df_01 = train_df.loc[:, ['ID', 'Alarm Name', 'Root Alarm']]
print("前十项: 'id，告警名称, 根告警'")
print(train_df_01[:10])
train_df_01['ID'] = train_df_01['ID'].astype(np.int)

train_id_input = train_df_01['ID'].values
train_name_input = train_df_01['Alarm Name'].values
train_root_input = train_df_01['Root Alarm'].values  # 类型numpy.ndarray
# train_root_input = list(train_root_input)
train_id_num = len(list(train_id_input.unique()))


# "ID + Alarm name"
# alarm_list, id_list：去重的id和alarm name列表，类型numpy.ndarray
alarm_list, id_list, net_input, net_input_01 = extractIDandAlarmname(train_dataset_path, IDSuccessive, binarizationFlag)
# 不重复的ID和alarm name数量，用于dictionary
n_alarmname, n_id = net_input.shape

# 将训练数据保存
outfile_dir, outfile_name = output_file(train_dataset_path, '_idAlarm_count')
net_input.to_csv(os.path.join(outfile_dir, outfile_name + '.csv'), header=True, index=False)

outfile_dir, outfile_name = output_file(train_dataset_path, '_idAlarm_bicount')
net_input_01.to_csv(os.path.join(outfile_dir, outfile_name + '.csv'), header=True, index=False)


nfeat = all_num_alarm  # 输入元素特征数，alarmname字典内的告警总数
nhid = 8  # GAT的隐层单元数
nclass = 1  # 输出类别数，降低最后输出(batch, nodes节点数，nclass输出特征数)，去除最后一个维度，使最终结果与节点数相关
in_drop = 0
coef_drop = 0
alpha = 0.2  # LeakyReLU的参数
nheads = 8  # attention的head数
num_lstm = 2  # 整个网络设计的LSTM网络个数
"""lstm隐层单元的个数"""
lstm_hidsize = 1
lstm_numlayers = 2  # 循环网络的LSTM层数
batch = 64  # batch_size

charlevel_lstm_gat = charlevel_BiLSTM_GAT(nfeat=nfeat, nhid=nhid, nclass=nclass, in_drop=in_drop, coef_drop=coef_drop,
                                          alpha=alpha, nheads=nheads, batch=batch, num_lstm=num_lstm,
                                          lstm_hidsize=lstm_hidsize, lstm_numlayers=lstm_numlayers)

# 损失函数
criterion = nn.NLLLoss()
learning_rate = 0.0001


def training(category_tensor, id_alarm_list):
    h_out, c_out = charlevel_lstm_gat.initHidden()

    charlevel_lstm_gat.zero_grad()

    for i in range(id_alarm_list.size()[0]):
        output, h_out, c_out = charlevel_lstm_gat(id_alarm_list[i], h_out, c_out)

    loss = criterion(output, category_tensor)
    loss.backward()

    # 将参数梯度加入value中（乘学习率）
    for p in charlevel_lstm_gat.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


# 网络预测的输出类型
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i


# 计时
print_every = 5000
plot_every = 1000


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

current_loss = 0
all_loss = []

# id_input, name_input分别为未去重的id和alarm name 数组，两数组的类型numpy.ndarray
# train_ids为训练集的id: alarm name list字典
train_ids = generate_ids_listi(train_id_input, train_name_input)
train_id_root = generate_ids_listi(train_id_input, train_root_input)
# category_tensor = torch.tensor([])

# 遍历id，alarmname列表的字典
for train_id, train_id_alarm_list in train_ids.items():
    id_alarm_tensor = siidToTensor(train_id_alarm_list, cols)
    category_name = train_id_root[train_id]
    category_index = alarmToIndex(category_name)
    # 格式为torch.Size([1]), 内容为tensor([i])
    category_tensor = torch.Tensor([category_index])

    output, loss = training(category_tensor, id_alarm_tensor)
    current_loss += loss

    if train_id % print_every == 0:
        guess_index = categoryFromOutput(output)
        correct = '√' if train_id_root[guess_index] == category_name else 'x (%s)' % category_name
        print('%d %d%% (%s) %.4f %s / %s %s' % (train_id, train_id / train_id_num * 100, timeSince(start),
                                                loss, train_id_alarm_list, train_id_root[guess_index], correct))

    # 添加当前平均损失
    if train_id == train_id_num:
        all_loss.append(current_loss / train_id_num)
        current_loss = 0

# 绘制损失函数图
plt.figure()
plt.plot(all_loss)






