import torch
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
train_dataset_path = "../labeled_june-train-sub5000.csv"

"""=================提取目前已知文件中所有的ID和alarm name================"""
dataset_path = "../labeled-June-10min(P)-RAN-20191023.csv"
df = pd.read_csv(dataset_path, header=0, index_col=None)

IDSuccessive = True  # 输入文件中ID是否连续（训练数据ID很可能不连续）
binarizationFlag = True  # 输出的矩阵是否为二值

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

# 总文件中出现的所有ID和alarmname的个数（去重后）
all_num_id = len(rows)
all_num_alarm = len(cols)

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

print('ids:')
print(str(ids))


# 确定alarm编号
def alarmToIndex(alarmname):
    return cols.find(alarmname)


# 将alarm转化为<1 * n_letters>张量(one-hot?)
def alarmToTensor(alarmname):
    tensor = torch.zeros(1, all_num_alarm)
    tensor[0][alarmToIndex(alarmname)] = 1
    return tensor


# 将line转换成<line_length*1 * n_letters>张量或者one-hot字母向量的数列
def idsToTensor(ids):
    tensor = torch.zeros(all_num_id, 1, all_num_alarm)
    k = 0
    for li, allist in enumerate(ids):
        for _, a in enumerate(allist):
            tensor[k][0][alarmToIndex(a)] = 1
            k += 1
    return tensor


# ID名称对应出现Alarm Name的计数矩阵（01矩阵为二值矩阵）
def extractIDandAlarmname(dfile, IDSuccessive, binarizationFlag):
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
alarm_list, id_list, net_input, net_input_01 = extractIDandAlarmname(df, IDSuccessive, binarizationFlag)
# 不重复的ID和alarm name数量，用于dictionary
n_alarmname, n_id = net_input.shape
print(n_alarmname, n_id)


"""===========================将名字转化为tensor类型==============================="""




