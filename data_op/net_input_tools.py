import pandas as pd
import numpy as np


def generate_input1(alarm_id_name, binarizationFlag=False):
    # binarizationFlag:处理结果文件保留二值性与否
    # 输入dataframe 包括ID, alarm name, ID从1开始连续编号，表示这一个片的
    # 输出每个ID对应的故障名出现的判断矩阵
    id_input = alarm_id_name['ID'].values
    name_input = alarm_id_name['Alarm Name'].values
    rows = alarm_id_name['ID'].unique()  # 去掉重复元素
    cols = alarm_id_name['Alarm Name'].unique()
    new_input = pd.DataFrame(np.zeros((len(rows), len(cols)), dtype=np.int), columns=cols)

    # zip(a,b): 将对象打包为元组
    for a, b in zip(id_input, name_input):
        new_input.at[a-1, b] = new_input.at[a-1, b] + 1
    if binarizationFlag:
        new_input[new_input>1] = 1

    return new_input


def generate_input2(alarm_id_name, binarizationFlag=False):
    # 输入dataframe 包括ID， alarm name, ID不从1开始编号
    id_input = alarm_id_name['ID'].values
    name_input = alarm_id_name['Alarm Name'].values
    rows = alarm_id_name['ID'].unique()
    cols = alarm_id_name['Alarm Name'].unique()
    new_input = pd.DataFrame(np.zeros((len(rows), len(cols)), dtype=np.int), columns=cols)

    lastID = None
    index = -1

    for a, b in zip(id_input, name_input):
        if lastID != a:
            index += 1
            lastID = a
        new_input.at[index, b] = new_input.at[index, b] + 1
    if binarizationFlag:
        new_input[new_input>1] = 1

    return new_input
