import torch

import pickle
import os
from model_BiLSTM_GAT import BiLSTM_GAT
import datetime
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # 不使用GPU


# 加载数据
train_dataset_path = "../labeled_june-train-sub5000.csv"