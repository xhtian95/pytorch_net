from gensim.models import Word2Vec
import os
import pandas as pd
import torch

import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

"""=================RNN====================="""
import torch.nn as nn

device = torch.device("cuda:0")


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i20 = nn.Linear(input_size+hidden_size, output_size)
        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        x = torch.cat((input, hidden), 1)
        hidden = self.i2h(x)
        output = self.i20(x)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


"""======================================读取文件数据，数据处理========================================"""
# dataset_path = ""../../../LSTM_gat_data/labeled-June-10min(P)-RAN-20191023.csv"
# dataset_path = "../../../LSTM_gat_data/labeled_june-train-sub5000.csv"
main_dataset_path = "../../../LSTM_gat_data/labeled_june.csv"
# my_data = pd.read_csv(dataset_path, header=0, index_col=None)


# 输出按序列去重后的数据列表及各个id中含有的告警数计数列表
def data_option(data_path, drop_duplicates=False, padd_seq_len=None):

    my_data = pd.read_csv(data_path, header=0, index_col=None)

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

    if padd_seq_len is None:
        # 各个id的告警计数写入对应的特征：dataset_target.groupby(by='ID').count()
        # .value_counts()：对所有alarmname出现次数值对应的次数记录（如有10个id都有5个告警，则对应的输出为 5 10）
        # 降序排列id内告警个数
        alarm_seq_len_distribution = dataset_target.groupby(by='ID').count()['Alarm Name'].value_counts().sort_index(
            ascending=False
        )
        # print(alarm_seq_len_distribution)
        # padding by max len，id内包含告警个数的最大值

    return dataset_target, alarm_seq_len_distribution


# 输出按序列去重后的数据列表及各个id中含有的告警数计数列表
dataset_target, alarm_seq_len_distribution = data_option(main_dataset_path, drop_duplicates=False,
                                                         padd_seq_len=None)
padd_seq_len = alarm_seq_len_distribution.index[0]


# 生成后续训练和验证所需的特征列表
def generate_feat_list(dataset_target):
    # alarm name的list集合
    alarm_name_list = []
    root_list = []
    # 字典中所有的alarm(去重，等价于求dictionary）
    singel_alarm_list = []
    # 所有的id列表
    ids_list = dataset_target['ID'].unique().tolist()

    # 按ID记录各个alarm的特征数据
    for eid in ids_list:
        cur_df = dataset_target[dataset_target['ID'] == eid]
        alarm_name_list.append(cur_df['Alarm Name'].values.tolist())
        root_list.append(cur_df['Root Alarm'].values.tolist()[0])  # 单个root alarm的类型为str

    singel_alarm_list = dataset_target['Alarm Name'].values.tolist()
    alarm_dic = list(set(singel_alarm_list))  # 去重

    return alarm_name_list, root_list, singel_alarm_list, ids_list, alarm_dic


# 生成后续训练和验证所需的特征列表
alarm_name_list, root_list, singel_alarm_list, ids_list, alarm_dic = generate_feat_list(dataset_target)

print("alarm dictionary")
print(alarm_dic[:50])
alarm_count = len(alarm_dic)

# """
print('ids_list[:20]')
print(ids_list[:20])
print('alarm_name_list[:20]')
print(alarm_name_list[:20])
print('root_list[:20]')
print(root_list[:20])
print('alarm_count:')
print(alarm_count)
# """


# 训练word2vec
word2vec_size = 200
"""
size：映射后向量的位数维数
sg={0,1}：  0：使用skip-gram模型；1：使用CBOW模型
"""
word2vec_model = Word2Vec(alarm_name_list, sg=1, size=word2vec_size, min_count=0)

"""
# 更新模型
model = Word2Vec.load(oldmodel_path)
model.build_vocab(new_list, update=True)
model.train(more_sentences, total_examples=model.corpus_count, compute_loss=True, epochs=)
"""


# 输入第ith个序列索引，全局告警名和根因列表，输出网络所需tensor及对应的category
# 将一条告警序列转换成<告警数 * 1 * word2vec_size>张量
# 根告警为<1 * word2vec_size>张量
def idlineToTensor(ith, alarm_name_list, root_list):
    # 第id个告警序列中的告警数量
    # print(len(alarm_name_list[ith]))
    ith_alarm = alarm_name_list[ith]
    tensor = torch.zeros(len(ith_alarm), 1, word2vec_size)

    for i, alarm in enumerate(ith_alarm):
        alarm_vec = word2vec_model.wv[alarm]
        alarm_tensor = torch.from_numpy(alarm_vec)
        # alarm_tensor = torch.Tensor(alarm_vec)
        tensor[i] = alarm_tensor

    root = root_list[ith]
    # 思考类型可能为alarm对应的vec
    category = torch.zeros(1, word2vec_size)

    cate = word2vec_model.wv[root]
    root_tensor = torch.from_numpy(cate)
    # root_tensor = torch.Tensor(cate)
    category[0] = root_tensor

    # 返回告警序列alarm name和root alarm对应的名字及word2vec向量
    return ith_alarm, root, tensor, category


_, _, alarm_name_tensor, category_tensor = idlineToTensor(0, alarm_name_list, root_list)


"""==================================training========================================"""
n_hidden = 128
rnn = RNN(word2vec_size, n_hidden, word2vec_size)

# 损失函数
# criterion = nn.NLLLoss()
criterion = nn.MSELoss()
learning_rate = 0.005


def training(category_tensor, id_alarm_list):
    h_out = rnn.initHidden()

    rnn.zero_grad()

    for i in range(id_alarm_list.size()[0]):
        output, h_out = rnn(id_alarm_list[i], h_out)
    # print(output.dtype)
    # print(category_tensor.dtype)
    # category_tensor = category_tensor.float()

    loss = criterion(output, category_tensor)
    loss.backward()

    # 将参数梯度加入value中（乘学习率）
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


def get_att_dis(target, alarm_dic):
    attention_distribution = []
    for alarm in alarm_dic:
        behaviored = torch.from_numpy(word2vec_model.wv[alarm])
        attention_score = torch.cosine_similarity(target.view(1, -1), behaviored.view(1, -1))  # 计算每个元素与给定元素的余弦相似度
        attention_distribution.append(attention_score)
    attention_distribution = torch.Tensor(attention_distribution)

    return attention_distribution / torch.sum(attention_distribution, 0)  # 标准化

"""
# 将网络输出的output（单个类别的特征向量）转化为各个类别的相似度向量
def netoutputSimilar(output, alarm_dic):
    # output_list = output.numpy().tolist()
    similar = torch.Tensor(1, len(alarm_dic))
    for i, alarm in enumerate(alarm_dic):
        similar[0][i] = torch.cosine_similarity(output, word2vec_model.wv[alarm])
    return similar
"""


# 网络预测的输出类型
def categoryFromOutput(output):
    # .topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
    # 沿给定的dim维度返回输入张量input中k个最大值
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# 开始训练
# 计时
print_every = 1000
plot_every = 1000

train_dataset_path = "../../../LSTM_gat_data/labeled_june-train.csv"
# 输出按序列去重后的数据列表及各个id中含有的告警数计数列表
train_target, train_alarm_seq_len_distribution = data_option(train_dataset_path, drop_duplicates=False,
                                                         padd_seq_len=None)
# 生成后续训练和验证所需的特征列表
train_alarm_name_list, train_root_list, singel_alarm_list, train_ids_list, _ = generate_feat_list(train_target)
# 告警id（序列）的数量
len_id = len(train_ids_list)

start = time.time()

current_loss = 0
all_loss = []
correct_num = 0

for ith, id in enumerate(train_ids_list):
    # 输入第ith个序列索引，全局告警名和根因列表，输出网络所需tensor及对应的category
    ith_alarm, ith_root, ith_alarm_tensor, ith_category_tensor = idlineToTensor(ith, train_alarm_name_list,
                                                                                train_root_list)

    output, loss = training(ith_category_tensor, ith_alarm_tensor)
    # print(type(output))
    # print(output.shape)
    # print(loss)
    current_loss += loss

    similarity = get_att_dis(output, alarm_dic)
    guess_index = categoryFromOutput(similarity)

    if alarm_dic[guess_index] == ith_root:
        correct_num += 1

    if ith % print_every == 0 and ith != 0:
        correct = '√' if alarm_dic[guess_index] == ith_root else 'x (%s)' % ith_root
        print('%d %d%% (%s) %.4f %s / %s %s' % (id, ith / len_id * 100, timeSince(start),
                                                loss, alarm_name_list[ith], alarm_dic[guess_index], correct))
        accuracy = correct_num / print_every * 100
        print("accuracy: ", accuracy)
        correct_num = 0

    if ith > 400:
        break

    # 添加当前平均损失，实时观测
    if ith / plot_every == 0:
        all_loss.append(current_loss / plot_every)
        current_loss = 0


# 绘制损失函数图
plt.figure()
plt.plot(all_loss)
plt.show()

"""=========================evaluate results=========================="""
confusion = torch.zeros(alarm_count, alarm_count)
n_confusion = 1000


# 不计算损失和参数变化， 其他和train过程相同
def evaluate(id_alarm_list):
    h_out = rnn.initHidden()

    rnn.zero_grad()

    for i in range(id_alarm_list.size()[0]):
        output, h_out = rnn(id_alarm_list[i], h_out)
    # print(output.dtype)
    # print(category_tensor.dtype)
    # category_tensor = category_tensor.float()

    return output


test_dataset_path = "../../../LSTM_gat_data/labeled_june-test.csv"
# 输出按序列去重后的数据列表及各个id中含有的告警数计数列表
test_target, test_alarm_seq_len_distribution = data_option(test_dataset_path, drop_duplicates=False,
                                                         padd_seq_len=None)
# 生成后续训练和验证所需的特征列表
train_alarm_name_list, train_root_list, singel_alarm_list, test_ids_list, _ = generate_feat_list(test_target)
len_test_id = len(test_ids_list)

correct_num = 0

for ith, id in enumerate(test_ids_list):
    # 输入第ith个序列索引，全局告警名和根因列表，输出网络所需tensor及对应的category
    ith_alarm, ith_root, ith_alarm_tensor, ith_category_tensor = idlineToTensor(ith, train_alarm_name_list,
                                                                                train_root_list)
    output = evaluate(ith_alarm_tensor)
    similarity = get_att_dis(output, alarm_dic)
    guess_index = categoryFromOutput(similarity)
    ith_label = alarm_dic.index(ith_root)
    # 横轴为label，纵轴为预测标签，出现即次数+1
    confusion[ith_label][guess_index] += 1

    if ith_label == guess_index:
        correct_num += 1

    if ith % print_every == 0 and ith != 0:
        accuracy = correct_num / ith
        print('%d %d%% (%s) accuracy = %.4f' % (id, ith / len_test_id * 100, timeSince(start), accuracy))
        accuracy = correct_num / print_every * 100
        print("accuracy: ", accuracy)
        correct_num = 0

    if ith > 400:
        break

# 取值
for i in range(alarm_count):
    confusion[i] = confusion[i] / confusion[i].sum()

# 建立plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# 建立axes坐标轴
ax.set_xticklabels([''] + alarm_dic, rotation=90)
ax.set_yticklabels([''] + alarm_dic)

# force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()


