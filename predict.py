import numpy as np

from model import *
from utils.snippets import DataGenerator, sequence_padding
from utils.tokenizers import Tokenizer

dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)
save_file_path = "{}/{}_tiny.h5".format(weights_path, MODEL_TYPE)
model = get_model()
model.load_weights(save_file_path)
tokenizer = Tokenizer(dict_path, do_lower_case = True)


class PredictGenerator(DataGenerator):
    """数据生成器
    """
    
    def __iter__(self, random = False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, text in self.sample(random):
            # print(text)
            token_ids, segment_ids = tokenizer.encode(text, maxlen = maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids]
                batch_token_ids, batch_segment_ids = [], []


def predict(text):
    batch_token_ids, batch_segment_ids = [], []
    token_ids, segment_ids = tokenizer.encode(text, maxlen = maxlen)
    batch_token_ids.append(token_ids)
    batch_segment_ids.append(segment_ids)
    y_pred = model.predict([batch_token_ids, batch_segment_ids])
    return np.argmax(y_pred)


def multi_predict(text_list):
    if len(text_list) > batch_size:
        raise "exceed the max num of predict sentenses batch"
    else:
        predict_generater = PredictGenerator(text_list)
        # for i, j in enumerate(predict_generater):
        #     print(j[0])
        pred_list = model.predict_generator(predict_generater.__iter__(), steps = len(predict_generater))
        # predict_generater.__iter__() 不打乱
        # predict_generater.forfit() 打乱
        return np.argmax(pred_list, axis = 0)


print(predict(
    "食管通过顺利，粘膜无殊。贲门齿状线清，粘膜无殊，胃底粘膜无殊，胃体粘膜充血，近胃窦部见一糜烂灶（活检2）。胃角胃窦部粘膜粗糙，红白相间，局部粘膜白稍多，见散在糜烂（活检1）。幽门孔圆，开放可。十二指肠球部粘膜无殊。食管通过顺利，粘膜无殊。贲门齿状线清，粘膜无殊，胃底粘膜无殊，胃体粘膜充血，近胃窦部见一糜烂灶（活检2）。胃角胃窦部粘膜粗糙，红白相间，局部粘膜白稍多，见散在糜烂（活检1）。幽门孔圆，开放可。十二指肠球部粘膜无殊。"
))
print(predict(
    "某患者，肛门出血，疑似痔疮"
))
print(multi_predict([
    "球囊扩张术，异物取出术",
    "慢性非萎缩性胃炎伴糜烂降部霜斑样糜烂",
    "所见回肠、结肠未见明显异常",
]))
