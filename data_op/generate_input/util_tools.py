from __future__ import unicode_literals, print_function, division
import pandas as pd
import os


# 拼接输入list中的元素
def concatDfFromDfList(input_df_list):
    df_concat_output = None
    for cur_df in input_df_list:
        if df_concat_output is None:
            df_concat_output = cur_df
        else:
            df_concat_output = pd.concat([df_concat_output, cur_df], axis=0)
    df_concat_output = df_concat_output.reset_index(drop=True)  # 重设索引
    return df_concat_output


# 获取所有的文件夹open_file_path内文件的路径
def get_all_path(open_file_path):
    rootdir = open_file_path
    path_list = []
    # 列出文件夹下所有的目录和文件
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        compath = os.path.join(rootdir, list[i])
        # print(compath)
        if os.path.isfile(compath):
            path_list.append(compath)
        if os.path.isdir(compath):
            path_list.extend(get_all_path(compath))
    return path_list


# 对标签数据里存在问题的siteID进行变换（改进）
def func_add0(x):
    if len(x) == 5:
        x = '0' + str(x)
    if x == 'eNodeB Test MM':
        x = 'ACXN'
    return x


# turn a unicode string to plain ascii
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# 获取alarmname的第一次出现的顺序{[alarm name]: 序号}
def getAlarmOrderDict(filepath=None):
    if filepath is None:
        return None
    else:
        with open(filepath, 'r') as f:
            header = f.readline()
    header = header.strip()
    alarmOrderDict = {}
    alarmNames = header.split(',')
    for index, item in enumerate(alarmNames):
        alarmOrderDict[item] = index + 1
    return alarmOrderDict

from collections import OrderedDict

if __name__=='__main__':
    # 文件为mmhc_input的csv
    a = getAlarmOrderDict("C:/Users/77037/PycharmProjects/untitled/" +
                          "labeled_june-01_new.csv")
    print(len(a), a)


"""related to word2vec"""
# 获取词对应索引及索引对应词的字典
class Lang:
    def __init__(self, name, vocab_list, SOS_token, EOS_token):
        """
        :param name:字典对应名称
        :param vocab_list: 字典列表
        :param SOS_token: 开始符
        :param EOS_token: 结束符
        """
        super(Lang, self).__init__()
        self.name = name
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.word2index = {}
        self.index2word = {}

        for index, word in enumerate(vocab_list):
            self.word2index[word] = index
            self.index2word[index] = word
        self.n_words = len(self.word2index)
