from util_tools import Lang
import torch
from gensim.models import Doc2Vec, Word2Vec
import pandas as pd
import numpy as np
import pickle


def word2vec_encoder(timesplit_path=None, word2vec_modelPath=None, encoder_modelPath=None, drop_duplicates=True,
                     padd_seq_len=None, adj_mat_alarm_dict_path=None):
    """
    :param timesplit_path: 切片文件路径
    :param word2vec_modelPath: 已存储的word2vec的模型路径
    :param encoder_modelPath:
    :param drop_duplicates: 是否去重
    :param padd_seq_len:
    :return:
    """
    with open(adj_mat_alarm_dict_path, 'rb') as f:
        alarm_graph, alarm2index = pickle.load(f)

    word2vec_model = Word2Vec.load(word2vec_modelPath)
    # 词向量存储在model.wv的KeyedVectors实例中，可以直接在KeyedVectors中查询词向量
    vocab_list = word2vec_model.wv.index2word  # 单词顺序
    # 获取字典列表
    emb_vocab_list = ['SOS', 'EOS'] + vocab_list  # 添加起始符和终止符

    # 获取词对应索引及索引对应词的字典
    # alarm_Lang属于类Lang
    alarm_Lang = Lang('alarm', emb_vocab_list, SOS_token=0, EOS_token=1)
    encoder = torch.load(encoder_modelPath)

    # 读取切片文件
    my_data = pd.read_csv(timesplit_path, header=0, index_col=None)
    # 过滤掉没有根因的
    dropId = set({})
    for egroup in my_data.groupby(by='ID'):
        if 1 not in egroup[1]['Label'].values:
            dropId.add(egroup[0])
    data = my_data[my_data['ID'].isin(dropId) == False]
    # data_target只包含接下来用到的列
    useful_cols = ['ID', 'Alarm Name', 'Label', 'Root Alarm']

    if drop_duplicates:
        # 去重
        dataset_target = data[useful_cols].drop_duplicates()
    else:
        dataset_target = data[useful_cols]

    dataset_target['Loss Weight'] = 1

    if padd_seq_len is None:
        # 各个id的告警计数写入对应的特征：dataset_target.groupby(by='ID').count()
        # .value_counts()：对所有alarmname出现次数值对应的次数记录（如有10个id都有5个告警，则对应的输出为 5 10）
        # 降序排列id内告警个数
        alarm_seq_len_distribution = dataset_target.groupby(by='ID').count()['Alarm Name'].value_counts().sort_index(
            ascending=False
        )
        # padding by max len，id内包含告警个数的最大值
        padd_seq_len = alarm_seq_len_distribution.index[0]
    print('alarm sequence length: ', padd_seq_len)

    # 从data_target生成训练数据
    X = []  # 由告警名称序列组成的列表
    Y = []  # 由根因标记0，1序列组成的列表
    Y_name = []
    loss_weight = []
    # 所有的id列表
    ids_list = dataset_target['ID'].unique().tolist()

    # 按ID记录各个alarm的特征数据
    for eid in ids_list:
        cur_df = dataset_target[dataset_target['ID'] == eid]
        X.append(cur_df['Alarm Name'].values)
        Y.append(cur_df['Label'].values.argmax())
        Y_name.append(cur_df['Root Alarm'].unique())
        loss_weight.append(cur_df['Loss Weight'].values[0])
    loss_weight = np.array(loss_weight)

    # 获取因果图的子图
    A = []
    for eachSeq in X:
        # 获取各个id的告警名称列表对应的索引（索引为列表形式）
        index_list = [alarm2index.get(item, 0) for item in eachSeq]
        if 0 in index_list:
            print("告警序列{}中含有alarm graph上不存在的告警{}！".format(eachSeq, index_list))
        son_graph = alarm_graph[index_list, :]
        son_graph = son_graph[:, index_list]
        # A即子图元素为各个id出现的告警之间的关系图（不考虑与其他告警之间的关系）
        A.append(son_graph)

    # word2index输出
    x_vec = []
    for item in X:
        temp_list = []
        for each_alarm_name in item:
            alarm_vec = get_vec(encoder, each_alarm_name, alarm_Lang)
            temp_list.append(alarm_vec)
        x_vec.append(np.array(temp_list))

    vec_len = encoder.hidden_size
    print("vec len: ", vec_len)

    x_vec_padd = np.zeros((len(x_vec), padd_seq_len, vec_len), dtype=np.float)
    a_padd = np.zeros((len(x_vec), padd_seq_len, padd_seq_len), dtype=np.float)
    for index in range(0, len(x_vec)):
        x_vec_padd[index, 0:x_vec[index].shape[0], :] = x_vec[index]
        a_padd[index, 0:A[index].shape[0], 0:A[index].shape[1]] = A[index]
    Y = np.array(Y)
    return [x_vec_padd, Y, a_padd, loss_weight, X, Y_name]


def get_vec(encoder, alarmname, lang):
    with torch.no_grad():
        # 获取所有（alarmname序列中）告警名对应的索引
        indexes = [lang.word2index[word] for word in alarmname.split(' ')]
        indexes.append(lang.EOS_token)  # 添加结束符
        # 将索引列表格式改为一列
        input_tensor = torch.tensor(indexes, dtype=torch.long).view(-1, 1)
        encoder_hidden = encoder.initHidden()
        input_length = input_tensor.size()[0]
        # 依次输入索引，编码器迭代训练
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # 编码器隐层的（0，0）列表为函数的输出
        vec = encoder_hidden[0][0].numpy().tolist()

        return vec