from keras import Model
from keras.layers import Lambda, Dense

from path import *
from utils.backend import search_layer, K
from utils.backend import set_gelu
from utils.models import build_transformer_model
from utils.optimizers import extend_with_piecewise_linear_lr, Adam

set_gelu('tanh')  # 切换gelu版本

num_classes = 3
maxlen = 200
batch_size = 128
config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME


def get_model():
    # 加载预训练模型
    bert = build_transformer_model(
        config_path = config_path,
        checkpoint_path = checkpoint_path,
        return_keras_model = False,
        model = MODEL_TYPE
    )
    
    output = Lambda(lambda x: x[:, 0], name = 'CLS-token')(bert.model.output)
    output = Dense(
        units = num_classes,
        activation = 'softmax',
        kernel_initializer = bert.initializer
    )(output)
    
    model = Model(bert.model.input, output)
    model.summary()
    
    # 派生为带分段线性学习率的优化器。
    # 其中name参数可选，但最好填入，以区分不同的派生优化器。
    AdamLR = extend_with_piecewise_linear_lr(Adam, name = 'AdamLR')
    
    model.compile(
        loss = loss_with_gradient_penalty,
        # loss = "sparse_categorical_crossentropy",
        # optimizer=Adam(1e-5),  # 用足够小的学习率
        optimizer = AdamLR(lr = 1e-4, lr_schedule = {
            1000: 1,
            2000: 0.1
        }),
        metrics = ['sparse_categorical_accuracy'],
        # metrics = ['accuracy'],
    )
    
    return model


def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


def loss_with_gradient_penalty(y_true, y_pred, epsilon = 1):
    """带梯度惩罚的loss
    """
    loss = K.mean(sparse_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, 'Embedding-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values ** 2)
    return loss + 0.5 * epsilon * gp
