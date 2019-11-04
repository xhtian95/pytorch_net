import torch.nn as nn
import numpy as np
import pandas as pd

import pickle
import os
from model_BiLSTM_GAT import BiLSTM_GAT
import datetime
import time
from generate_input import util_tools
from net_input_tools import *

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
rows = df_1['ID'].unique()  # 去掉重复元素
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


# 将每个id出现的alarm列表转换成<list_length * 1 * n_letters>张量或者one-hot字母向量的数列
def siidToTensor(idlist, nameIndexDic):
    tensor = torch.zeros(len(idlist), 1, all_num_alarm)
    for li, alarm in enumerate(idlist):
        tensor[li][0][alarmToIndex(alarm, nameIndexDic)] = 1
    return tensor


"""idsToTensor为全部alarm的tensor"""
# 将line转换成<line_length * 1 * n_letters>张量或者one-hot字母向量的数列
# nameIndexDic为原始去重alarm name的列表，以此获得纵坐标（对应alarm name的索引）
def idsToTensor(iddic, nameIndexDic):
    tensor = torch.zeros(entire_num_list, 1, all_num_alarm)
    pddataframe = pd.DataFrame(np.zeros((entire_num_list, all_num_alarm), dtype=np.int), columns=cols)
    k = 0
    for allist in iddic.values():
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
outfile_dir, outfile_name = output_file(dataset_path, '_tensor')
print(tensor_ids.shape)
# dataframe.to_csv(os.path.join(outfile_dir, outfile_name + ".csv"), header=True, index=False)


"""===========================ID名称对应出现Alarm Name的计数==============================="""

IDSuccessive = False  # 输入文件中ID是否连续（训练数据ID很可能不连续）
binarizationFlag = True  # 输出的矩阵是否为二值

train_df = pd.read_csv(train_dataset_path, header=0, index_col=None)


# ID名称对应出现Alarm Name的计数矩阵（01矩阵为二值矩阵）
def extractIDandAlarmname(dfile, IDSuccessive, binarizationFlag=True):
    df_1 = dfile.loc[:, ['ID', 'Alarm Name', 'Root Alarm']]
    # print("前十项: 'id，告警名称, 根告警'")
    # print(df_1[:10])
    df_1['ID'] = df_1['ID'].astype(np.int)

    if IDSuccessive:
        alarm_list, id_list, net_input = generate_input3(df_1)
        _, _, net_input_01 = generate_input3(df_1, binarizationFlag=True)
    else:
        alarm_list, id_list, net_input = generate_input4(df_1)
        _, _, net_input_01 = generate_input4(df_1, binarizationFlag=True)

    return alarm_list, id_list, net_input, net_input_01


# "ID + Alarm name"
alarm_list, id_list, net_input, net_input_01 = extractIDandAlarmname(train_dataset_path, IDSuccessive, binarizationFlag)
# 不重复的ID和alarm name数量，用于dictionary
n_alarmname, n_id = net_input.shape

outfile_dir, outfile_name = output_file(dataset_path, '_idAlarm_count')
net_input.to_csv(os.path.join(outfile_dir, outfile_name + '.csv'), header=True, index=False)

outfile_dir, outfile_name = output_file(dataset_path, '_idAlarm_bicount')
net_input_01.to_csv(os.path.join(outfile_dir, outfile_name + '.csv'), header=True, index=False)


"""==================================training========================================"""


# 损失函数
criterion = nn.NLLLoss()
learning_rate = 0.0001

def training(category_tensor, alarm_list_tensor):



