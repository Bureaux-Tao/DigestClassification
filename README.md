# 上下消化道病历分类

通过ALBERT_tiny来对病历文本自动进行分类，并总结出输入词的特征重要性表。

其实本可以找一堆代表上下消化道的关键词来进行字符串搜索进行判断，也可以取得不错的accurency，但是不能涵盖全部，且比较繁琐，泛化性不强。通过分类网络，可以学习到哪些特征比较重要，在完成分类的同时也可以总结出关键词表。

## Dataset

数据集包含肠镜、胃镜等常规检查的文本数据集，共2636条。由于任务简单、主要是依靠文本局部特征来分类，不怎么需要用到前后文语义特征，故数量也是足够的。

数据集形如：

```
慢性非萎缩性胃炎，伴糜烂伴胆汁返流	1
肛门通过顺利，戴透明帽进镜，距肛缘65cm肝曲处可见一亚蒂息肉，大小约0.5*0.6cm，予圈套摘除，创面予钛夹一枚夹闭。	2
“横结肠”及“直肠”管状腺瘤伴低级别上皮内瘤变。“降结肠”增生性息肉。	2
胆总管结石ERCP+EST胆道塑料支架置入术	0
```

标签中：

- 1表示文本在描述上消化道
- 2表示文本在描述下消化道
- 0表示文本既不描述上消化道也不描述下消化道

数据集经过筛选，最长长度200字符。可以换成自己的数据集，但要注意：标签序号必须从0开始，到`model.py`中的`num_classes-1`为止

## Steps

1. 替换数据集
2. 修改model.py中的`num_classes`、`maxlen`、`batch_size`
3. 下载权重，放到项目中
4. 修改path.py中的地址
5. 根据需要修改model.py模型结构和超参数
6. 训练前debug看下train_generator数据
7. 训练

## Project Structure

```
./
├── README.md
├── albert_tiny_google_zh                       ALBERT权重
│   ├── albert_config.json
│   ├── albert_model.ckpt.data-00000-of-00001
│   ├── albert_model.ckpt.index
│   ├── checkpoint
│   └── vocab.txt
├── continue_fit.py                             中途断了继续训练
├── data                                        数据集
│   ├── all.txt
│   ├── test.txt
│   ├── train.txt
│   ├── val.txt
├── data_process.py                             数据加载
├── feature_importance.py                       从训练好的模型中提取特征重要性
├── images                                      输入特征热力图样本
│   ├── sample1.png
│   ├── sample2.png
│   └── sample3.png
├── log                                         训练日志
│   ├── nohup.out
├── main.py                                     训练
├── model.py                                    模型
├── path.py                                     所有路径
├── predict.py                                  预测与批预测
├── preprocess                                  数据预处理
│   ├── __init__.py
│   ├── data_transfer.py
│   └── generate4label.py
├── statistics.py                               数据统计
├── utils                                       bert4keras包，也可以pip下载
│   ├── __init__.py
│   ├── __pycache__
│   ├── adversarial.py
│   ├── backend.py
│   ├── layers.py
│   ├── models.py
│   ├── optimizers.py
│   ├── plot.py
│   ├── snippets.py
│   └── tokenizers.py
└── weights                                     权重
    └── albert_tiny.h5
```

## Requirements

```
Keras==2.2.4
matplotlib==3.4.0
pandas==1.2.3
seaborn==0.11.2
tensorflow==1.14.0
tqdm==4.61.2
```

## Model

ALBERT权重使用经过转换brightmart版的albert权重。[下载连接](https://github.com/bojone/albert_zh)

可以根据需要换成albert base、albert large、albert xlarge。

转换出来的模型，Embedding层都是没有低秩分解的，但是保留了transformer block的跨层参数共享。

> ALBert is based on Bert, but with some improvements. It achieves state of the art performance on main benchmarks with 30% parameters less.
> For albert_base_zh it only has ten percentage parameters compare of original bert model, and main accuracy is retained.
>
> Different version of ALBERT pre-trained model for Chinese, including TensorFlow, PyTorch and Keras, is available now.
>
> 海量中文语料上预训练ALBERT模型：参数更少，效果更好。预训练小模型也能拿下13项NLP任务，ALBERT三大改造登顶GLUE基准
>
> 一键运行10个数据集、9个基线模型、不同任务上模型效果的详细对比，见中文语言理解基准测评 CLUE benchmark

模型大小

> albert_tiny_zh, albert_tiny_zh(训练更久，累积学习20亿个样本)，文件大小16M、参数为4M
>
> albert_tiny使用同样的大规模中文语料数据，层数仅为4层、hidden size等向量维度大幅减少; 尝试使用如下学习率来获得更好效果：{2e-5, 6e-5, 1e-4}
>
> albert_tiny_google_zh(累积学习10亿个样本,google版本)，模型大小16M、性能与albert_tiny_zh一致
>
> albert_small_google_zh(累积学习10亿个样本,google版本)， 速度比bert_base快4倍；LCQMC测试集上比Bert下降仅0.9个点；去掉adam后模型大小18.5M
>
> albert_large_zh,参数量，层数24，文件大小为64M
>
> albert_base_zh(额外训练了1.5亿个实例即 36k steps * batch_size 4096); albert_base_zh(小模型体验版), 参数量12M, 层数12，大小为40M
>
> 参数量为bert_base的十分之一，模型大小也十分之一；在口语化描述相似性数据集LCQMC的测试集上相比bert_base下降约0.6~1个点；相比未预训练，albert_base提升14个点
>
> albert_xlarge_zh_177k ; albert_xlarge_zh_183k(优先尝试)参数量，层数24，文件大小为230M
>
> 参数量和模型大小为bert_base的二分之一；需要一张大的显卡；完整测试对比将后续添加；batch_size不能太小，否则可能影响精度


## Train

将训练样本按8：1：1划分成训练集、验证集、测试集，并shuffle。

使用warmup调整学习率，0～1000步时，学习率从0线性增加到1e-4，然后1000～2000步时，学习率从1e-3线性下降到1e-5，2000步后学习率保持1e-5不变。

监控val_loss，val_loss 3轮不降即停。

```
Epoch 1/200
29/29 [==============================] - 19s 643ms/step - loss: 1.1703 - acc: 0.2036 - val_loss: 1.0676 - val_acc: 0.4705
val_acc: 0.47046, best_val_acc: 0.47046, test_acc: 0.49440

Epoch 2/200
29/29 [==============================] - 11s 374ms/step - loss: 1.1187 - acc: 0.3962 - val_loss: 1.0310 - val_acc: 0.4551
val_acc: 0.45514, best_val_acc: 0.47046, test_acc: 0.48101

Epoch 3/200
29/29 [==============================] - 10s 361ms/step - loss: 1.0548 - acc: 0.4824 - val_loss: 1.0004 - val_acc: 0.4530
val_acc: 0.45295, best_val_acc: 0.47046, test_acc: 0.48046

... ...

Epoch 23/200
29/29 [==============================] - 13s 458ms/step - loss: 0.0211 - acc: 0.9968 - val_loss: 0.0175 - val_acc: 0.9956
val_acc: 0.99562, best_val_acc: 0.99562, test_acc: 0.99699

Epoch 24/200
29/29 [==============================] - 13s 442ms/step - loss: 0.0190 - acc: 0.9966 - val_loss: 0.0179 - val_acc: 0.9956
val_acc: 0.99562, best_val_acc: 0.99562, test_acc: 0.99699

Epoch 25/200
29/29 [==============================] - 13s 459ms/step - loss: 0.0184 - acc: 0.9962 - val_loss: 0.0182 - val_acc: 0.9956
val_acc: 0.99562, best_val_acc: 0.99562, test_acc: 0.99699

Epoch 00025: early stopping
```

## Evaluate

本项目并未计算F1，仅acc，读者可以自行计算每类F1值

```
final test acc: 0.996994
```

## Predict

### 单条预测

```
predict(
    "食管通过顺利，粘膜无殊。贲门齿状线清，粘膜无殊，胃底粘膜无殊，胃体粘膜充血，近胃窦部见一糜烂灶（活检2）。胃角胃窦部粘膜粗糙，红白相间，局部粘膜白稍多，见散在糜烂（活检1）。幽门孔圆，开放可。十二指肠球部粘膜无殊。食管通过顺利，粘膜无殊。贲门齿状线清，粘膜无殊，胃底粘膜无殊，胃体粘膜充血，近胃窦部见一糜烂灶（活检2）。胃角胃窦部粘膜粗糙，红白相间，局部粘膜白稍多，见散在糜烂（活检1）。幽门孔圆，开放可。十二指肠球部粘膜无殊。"
)
```

结果：`1`

### 批量预测

```
multi_predict([
    "球囊扩张术，异物取出术",
    "慢性非萎缩性胃炎伴糜烂降部霜斑样糜烂",
    "所见回肠、结肠未见明显异常",
])
```

结果：`[0, 1, 2]`

ATTENTION: 批量预测最多512条

## 输入文本特征重要性

通过积分梯度[Sundararajan M, Taly A, Yan Q. Axiomatic Attribution for Deep Networks[J]. arXiv pre-print server, 2017.](https://arxiv.org/abs/1312.6034)来提取

读取train or val or test文件，遍历每条，每条的每个字都会计算出一个score值，0 ≤ socre ≤ 1，然后设置一个门限值，将score大于该门限值的字，或多个字连在一起组成的词抽取出来，放入一个数组，最后统计该字or词出现的frequency，排序。

### 步骤

1. 替换`feature_importance.py`中的`save_file_path`模型权重
2. 替换`feature_importance.py`中的`threshhold`门限值
3. 如果仅抽取每条中大于门限值的最大值，`extracts_max`设为`True`，如果提取每个大于门限值的字词，设为`False`
4. 如果要画出注意力热力图，解开`plot_attention(data)`的注释，但是速度会慢很多

### 热力图样例

![](images/sample1.png)

![](images/sample2.png)

![](images/sample3.png)

### 运行输出

当门限值设为0.2，提取所有大于0.2的所有字词时

```
[
    ('胃', 2282), 
    ('肠', 1497), 
    ('食管', 807), 
    ('肛门', 541), 
    ('胃炎', 523), 
    ('殊。胃', 487),
    
    ...
    
    ('、升结肠', 1), 
    ('盲肠、升结肠见', 1), 
    ('。肝', 1), 
    ('退镜距肛', 1)
]
```

前几位的确是上下消化道的特征，说明模型有效。

读者还可以根据类别对其进行分类。