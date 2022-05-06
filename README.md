# 日志语言模型

这个仓库是基于HuggingFace Transformers库搭建的预训练语言模型，选择的是基于英文语料进行预训练的 bert-base-uncased 模型。

该仓库可根据日志语料库构建用于 MLM 和 NSP 预训练任务的训练集，在 bert-base-uncased 预训练参数的基础上进行微调，得到可用于提取日志模板向量的日志语言模型。

该仓库中共包含两种语言模型构建方案：直接对 bert 预训练模型进行微调或训练 K-Adapters 进行特征融合。

参考文献：[BERT](https://arxiv.org/abs/1810.04805) , [K-Adapters](https://arxiv.org/abs/2002.01808)

## 环境安装

+ python
+ pytorch
+ transformers

## 代码结构说明

+ #### **Model**

  文件夹中包含语言模型神经网络及其组件代码。

  + **Bert.py**

    包含可进行 MLM 和 NSP 任务的预训练语言模型 **BertForPreTraining** 类以及加载训练集所需的 Dataset 和 DataLoader

  + **Adapter.py**

    包含可嵌入到 bert 模型中根据部分层的输出向量进行训练的 **Adapter** 类，Adapter模型结构如下图所示：

    ![adapter](.\image\adapter.jpg)

    Adapter 由两个线性投影层和一个N层的Transformer encoder 组成， 代码中 N=2，隐藏层维度为768。

  + **AdapterModel.py**

    包含两个类，**AdapterModel** 类由固定参数的 bert 预训练模型和 k个Adapter组建而成，代码中默认k=3，分别接收第0，6，11层bert encoder的输出向量，最后 bert 预训练模型的输出向量和最后一个adapter的输出向量合并后通过一个全连接层得到输入序列的编码向量。该模型在经过日志语料微调后将参数保存，可用于日志模板向量的提取。

    **AdapterForPretraining** 类将 AdapterModel 的输出向量输入到解码器中完成 MLM 和 NSP 任务的训练。

+ #### **create_pretraining_data.py**

  根据输入日志语料构建预训练任务的训练集。

  输入参数要求：

  + **input_file**：**必填**，日志语料文件，其中每条日志语句占一行，同一个序列中的日志语句为一个段落，段落间以空行分隔。
  + **output_dir**: **必填**，训练集和验证集输出目录，默认文件名称：log_train_file, log_valid_file。包含5个字段：
    + **input_ids**: 语句经过分词和mask后对应的词片索引
    + **segment_ids**: 0代表第一句，1代表第二句
    + **attention_mask**: 是否需要计算注意力，1代表计算注意力
    + **labels**: 被mask的词片对应的原词索引，-100代表未被 mask
    + **next_sentence_label**: 句子对是否连续，1代表不连续，0代表连续

  + **max_seq_length**：**可选**，最大序列长度，默认128，小于此长度会进行填充。
  + **masked_lm_prob**: **可选**，序列中需要被覆盖的词汇比例，默认0.15
  + **max_predictions_per_seq**：**可选**，序列中最多被覆盖的词汇数，默认为20。
  + **dupe_factor**：**可选**，对同一语料进行多次覆盖操作，增加训练集数量，提高模型学习效果，默认为10

+ #### **run_pretraining.py**

  使用训练集对预训练语言模型进行微调，得到日志语言模型。

  输入参数要求：

  + **train_file**: **必填**，训练集文件地址
  + **valid_file**: **必填**，验证集文件地址
  + **output_dir**: **必填**，模型存放地址，输出包含pytorch_model.bin, config.json两个文件
  + **do_train**：**可选**，是否进行训练，默认为True
  + **do_eval**: **可选**， 是否进行验证，默认为True

+ #### adapter_pretraining.py

  使用训练集训练含有Adapter的预训练语言模型。

  输入参数要求：

  + **train_file**: **必填**，训练集文件地址
  + **valid_file**: **必填**，验证集文件地址
  + **output_dir**: **必填**，Adapter 模型存放地址，输出包含pytorch_model.bin, config.json两个文件
  + **do_train**：**可选**，是否进行训练，默认为True
  + **do_eval**: **可选**， 是否进行验证，默认为True
  + **adapter_transformer_layers**: **可选**，一个Adapter中Transformer encoder层数，默认为2
  + **adapter_size**: **可选**，adapter隐藏层维度，默认768
  + **adapter_list**: **可选**，选择 bert 模型中部分隐藏层的输入用于训练Adapter，默认0，6，11。

+ #### feature_extraction.py

  使用微调后的语言模型提取日志模板向量。

  输入参数要求：

  + **input_file**：**必填**，日志模板文件，这里输入的是json格式文件，包含异常检测中日志模板索引及其对应的用于词向量加权求平均的方案中对应的日志模板词序列。
  + **model_path**：**必填**，日志语言模型参数所在文件夹，对应于上两个文件的output_dir。
  + **output_dir**: **必填**, 日志模板向量存放文件，格式与 LogAnomaly 和 LogRobust 中存放模板向量的文件格式一致。

## 使用方法与注意事项

先运行create_pretraining_data.py, 然后运行 run_pretraining.py 或 adapter_pretraining.py得到不同语言模型，最后运行feature_extraction.py 提取模板向量即可。

注：所需的外部文件有两个，日志语料文件：语料处理方法根据数据集变化，保证语句和段落的格式为语句单独一行，段落间空行相隔即可；日志模板文件：包含所有日志模板语句即可，若格式不同需更改文件解析代码。

在Data文件夹中给出我使用的阿里云比赛日志对应的两个外部文件。

另外在运行 crate_pretraining_data.py 时，tokenizer会给出序列过长的警告，后续在构建数据集时会修剪序列长度，所以无需理会。



