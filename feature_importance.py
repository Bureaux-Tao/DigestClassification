#! -*- coding: utf-8 -*-
# 通过积分梯度（Integrated Gradients）来给输入进行重要性排序
# 接 task_sentiment_albert.py
# 原始论文：https://arxiv.org/abs/1703.01365
# 博客介绍：https://kexue.fm/archives/7533
# 请读者务必先弄懂原理再看代码，下述代码仅是交互式演示代码，并非成品API

##
import pickle
from collections import Counter

import numpy as np
import seaborn as sns
from keras.layers import Layer, Input
from keras.models import Model
from matplotlib import pyplot as plt
from tqdm import tqdm

from model import get_model
from path import *
from utils.backend import K, batch_gather
from utils.tokenizers import Tokenizer

##

threshhold = 0.5
extracts_max = True

dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)
save_file_path = "{}/{}_tiny_no_dump.h5".format(weights_path, MODEL_TYPE)
model = get_model()
model.load_weights(save_file_path)
tokenizer = Tokenizer(dict_path, do_lower_case = True)


##

class Gradient(Layer):
    """获取梯度的层
    """
    
    def __init__(self, **kwargs):
        super(Gradient, self).__init__(**kwargs)
        self.supports_masking = True
    
    def call(self, input):
        input, output, label = input
        output = batch_gather(output, label)
        return K.gradients(output, [input])[0] * input
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]


def plot_attention(tpl):
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['savefig.dpi'] = 2000  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    tokens = []
    values = []
    for t, v in tpl:
        tokens.append(t)
        values.append(v)
    # creating random data
    plt.rcParams["figure.figsize"] = 60, 1
    data = np.array([values])
    text = np.array([tokens])
    
    # combining text with values
    formatted_text = (np.asarray(["{0}".format(text) for text in text.flatten()])).reshape(data.shape[0], data.shape[1])
    
    # drawing heatmap
    plt.subplots()
    sns.heatmap(data, annot = formatted_text, fmt = "", cmap = "cool")
    plt.axis('off')
    plt.show()


label_in = Input(shape = (1,))  # 指定标签
input = model.get_layer('Embedding-Token').output
output = model.output
grads = Gradient()([input, output, label_in])
grad_model = Model(model.inputs + [label_in], grads)

# 获取原始embedding层
embeddings = model.get_layer('Embedding-Token').embeddings
values = K.eval(embeddings)

all = []
extract = []
with open('./data/all_no_dump.txt', 'r', encoding = 'utf-8') as f:
    for i in f.readlines():
        all.append(i.strip("\n").split('\t')[0])

for text in tqdm(all):
    # text = u'透明帽辅助下进镜，食道距门齿30cm见一鱼刺，予异物钳钳夹取出，过程顺利。'
    token_ids, segment_ids = tokenizer.encode(text)
    preds = model.predict([[token_ids], [segment_ids]])
    label = np.argmax(preds[0])
    
    pred_grads = []
    n = 20
    for i in range(n):
        # nlp任务中参照背景通常直接选零向量，所以这里
        # 让embedding层从零渐变到原始值，以实现路径变换。
        alpha = 1.0 * i / (n - 1)
        K.set_value(embeddings, alpha * values)
        pred_grad = grad_model.predict([[token_ids], [segment_ids], [[label]]])[0]
        pred_grads.append(pred_grad)
    
    # 然后求平均
    pred_grads = np.mean(pred_grads, 0)
    
    # 这时候我们得到形状为(seq_len, hidden_dim)的矩阵，我们要将它变换成(seq_len,)
    # 这时候有两种方案：1、直接求模长；2、取绝对值后再取最大。两者效果差不多。
    scores = np.sqrt((pred_grads ** 2).sum(axis = 1))
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    scores = scores.round(4)
    results = [(tokenizer.decode([t]), s) for t, s in zip(token_ids, scores)]
    
    # scores = np.abs(pred_grads).max(axis = 1)
    # scores = (scores - scores.min()) / (scores.max() - scores.min())
    # scores = scores.round(4)
    # results = [(tokenizer.decode([t]), s) for t, s in zip(token_ids, scores)]
    # print(results[1:-1])
    
    data = results[1:-1]
    # plot_attention(data)
    max_token, max_score = max(data, key = lambda t: t[1])
    word = ""
    flag = False
    contains_max = False
    for token, value in data:
        if value >= threshhold:
            word += token
            flag = True
            if token == max_token and value == max_score:
                contains_max = True
        else:
            if flag == True:
                if extracts_max:
                    if contains_max == True:
                        extract.append(word)
                        break
                else:
                    extract.append(word)
                word = ""
                flag = False
            else:
                word = ""
                flag = False
    # print(extract)

##
with open("./extract_2.pkl", 'wb') as f1:
    pickle.dump(extract, f1)
freq = dict(Counter(extract))
print(sorted(freq.items(), key = lambda d: d[1], reverse = True))
