#-*- coding:utf-8 -*-
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def train_test_split_op(df,train_ratio=0.8, stratifyFlag = True,random_state=None):
    # 去除重复项和缺失值,得到不重复的ID Root Alarm对应值以及序列号(序列号，ID，Root Alarm)
    temp_df = df[["ID", "Root Alarm"]].dropna().drop_duplicates()

    if stratifyFlag == True:  # 设置为True的话，可以在类别不平衡的时候尽量保证分布
        # train_test_split(X, y, test_size, random_state)
        # X：待划分样本特征集合； y：带划分样本标签
        train_ids, test_ids, _, _ = train_test_split(temp_df["ID"], temp_df["Root Alarm"], test_size=1 - train_ratio,
                                                     random_state=random_state, stratify=temp_df["Root Alarm"])
    else:
        train_ids, test_ids, _, _ = train_test_split(temp_df["ID"], temp_df["Root Alarm"], test_size=1 - train_ratio,
                                                    random_state=random_state)
    train_ids = train_ids.values
    test_ids = test_ids.values
    df_train = df[df["ID"].isin(train_ids)]
    df_test = df[df["ID"].isin(test_ids)]
    return df_train,df_test


if __name__ == "__main__":
    # data_path = "/root/workspace/alarm_data/labeled_june.csv"
    data_path = "C:/Users/77037/PycharmProjects/untitled/labeled_june.csv"
    dir_name = os.path.split(data_path)[0]
    out_name = os.path.split(data_path)[1]
    out_name = out_name[:out_name.rfind(".")] # 不包括后缀名,去掉"."后的扩展名
    df = pd.read_csv(data_path,header=0,index_col=None)
    train_ratio=0.8
    df_train,df_test=train_test_split_op(df,train_ratio=train_ratio,stratifyFlag=True,random_state=33)
    # 导出csv文件
    df_train.to_csv(os.path.join(dir_name,out_name+"-train"+".csv"),header=True,index=False)
    df_test.to_csv(os.path.join(dir_name,out_name+"-test"+".csv"),header=True,index=False)

    # ids_list = df["ID"].dropna().unique().tolist()
    # if shuffleData == True:# shuffleData
    #     import random
    #     random.shuffle(ids_list) #打乱时间片
    # train_index=int(len(ids_list)*train_ratio)
    # train_ids = ids_list[:train_index]
    # test_ids = ids_list[train_index:]
    #
    # df_train = df[df["ID"].isin(train_ids)]
    # df_test = df[df["ID"].isin(test_ids)]