#! -*- coding:utf-8 -*-
# 情感分析例子，加载albert_zh权重(https://github.com/brightmart/albert_zh)

from model import *
from utils.adversarial import adversarial_training

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.callbacks import EarlyStopping, ModelCheckpoint

from utils.backend import set_gelu
from utils.tokenizers import Tokenizer
from utils.snippets import sequence_padding, DataGenerator
from utils.snippets import open

from path import BASE_MODEL_DIR, train_file_path, \
    val_file_path, test_file_path, MODEL_TYPE, weights_path

set_gelu('tanh')  # 切换gelu版本
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)
save_file_path = "{}/{}_tiny_1.h5".format(weights_path, MODEL_TYPE)
save_new_path = "{}/{}_tiny_continue_1.h5".format(weights_path, MODEL_TYPE)


def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename, encoding = 'utf-8') as f:
        for l in f:
            text, label = l.strip().strip('\n').split('\t')
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data(train_file_path)
valid_data = load_data(val_file_path)
test_data = load_data(test_file_path)

train_data.extend(valid_data)
train_data.extend(test_data)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case = True)


class data_generator(DataGenerator):
    """数据生成器
    """
    
    def __iter__(self, random = False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen = maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)

model = get_model()
model.load_weights(save_file_path)
adversarial_training(model, 'Embedding-Token', 0.5)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis = 1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


if __name__ == '__main__':
    save_model = ModelCheckpoint(save_new_path, monitor = 'sparse_categorical_accuracy', verbose = 0,
                                 mode = 'max', save_weights_only = True, save_best_only = True)
    early_stopping = EarlyStopping(monitor = 'sparse_categorical_accuracy', patience = 5, verbose = 1,
                                   mode = 'max')  # 提前结束
    
    # for i, item in enumerate(train_generator):
    #     print("\nbatch_token_ids shape: ", item[0][0].shape)
    #     print("batch_segment_ids shape:", item[0][1].shape)
    #     print("batch_labels shape:", item[1].shape)
    #     if i == 5:
    #         break
    
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch = len(train_generator),
        epochs = 200,
        callbacks = [early_stopping, save_model]
    )

else:
    model.load_weights(save_file_path)
