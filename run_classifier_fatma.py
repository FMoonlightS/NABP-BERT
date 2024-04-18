# coding:utf-8
import modeling
import tokenization
from run_classifier import create_model, file_based_input_fn_builder
import tensorflow as tf
import optimization
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, \
    confusion_matrix, roc_curve, average_precision_score
import os
import time

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string(
#    "init_checkpoint", None,
#    "Initial checkpoint (usually from a pre-trained BERT model).")

# Nb_Ag_1_1 (NoPreTrain)
#tf.app.flags.DEFINE_string(
#    'save_path', "./model/3kmer_Classifier_model_512/NbAg/NoPreTrain_1_4/model_NbAg_1.ckpt", '模型保存位置')

tf.app.flags.DEFINE_boolean('do_eval', True, '是否在训练后评估')
tf.app.flags.DEFINE_boolean('do_save_model', True, '是否在训练后保存模型')
# tf.app.flags.DEFINE_string('data_name', 'PPI', '导入的数据名')
tf.app.flags.DEFINE_string('data_name', 'NbAg', '导入的数据名')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch大小')
tf.app.flags.DEFINE_integer('num_train_epochs', 100, '训练的轮次')
tf.app.flags.DEFINE_float('warmup_proportion', 0.1, '预热的训练比例')
tf.app.flags.DEFINE_float('learning_rate', 2e-5, '学习率')
tf.app.flags.DEFINE_boolean('using_tpu', False, '是否使用TPU')
tf.app.flags.DEFINE_float('seq_length', 512, '序列长度')

# PPI
# tf.app.flags.DEFINE_string(
#     'data_root', './dataAfterPreProcessing/PPI_Dataset/PreProcessing_2/asTF_Record_2/', '使用数据集的根目录')

# Nb_Ag
tf.app.flags.DEFINE_string(
   'data_root', './dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/PreProcessing_2/asTF_Record/', '使用数据集的根目录')

tf.app.flags.DEFINE_string('vocab_file', './vocab/vocab_3kmer.txt', '词典目录')

tf.app.flags.DEFINE_string('bert_config', "./bert_config_3.json", 'bert配置')

# Pretrained Model 1_1
tf.app.flags.DEFINE_string(
    'init_checkpoint', "./model/3kmer_model/num_hidden_layers_1/num_attention_heads_1/model.ckpt-1000000", '模型的初始化节点')

# Pretrained Model 1_2
#tf.app.flags.DEFINE_string(
#    'init_checkpoint', "./model/3kmer_model/num_hidden_layers_1/num_attention_heads_2/model.ckpt-1000000", '模型的初始化节点')

# # Pretrained Model 1_3
#tf.app.flags.DEFINE_string(
#    'init_checkpoint', "./model/3kmer_model/num_hidden_layers_1/num_attention_heads_3/model.ckpt-1000000", '模型的初始化节点')

# # Pretrained Model 1_4
#tf.app.flags.DEFINE_string(
#    'init_checkpoint', "./model/3kmer_model/num_hidden_layers_1/num_attention_heads_4/model.ckpt-1000000", '模型的初始化节点')

# # Pretrained Model 1_6
#tf.app.flags.DEFINE_string(
#    'init_checkpoint', "./model/3kmer_model/num_hidden_layers_1/num_attention_heads_6/model.ckpt-1000000", '模型的初始化节点')

# # Pretrained Model 1_8
#tf.app.flags.DEFINE_string(
#    'init_checkpoint', "./model/3kmer_model/num_hidden_layers_1/num_attention_heads_8/model.ckpt-1000000", '模型的初始化节点')

# # Pretrained Model 1_12
#tf.app.flags.DEFINE_string(
#    'init_checkpoint', "./model/3kmer_model/num_hidden_layers_1/num_attention_heads_12/model.ckpt-1000000", '模型的初始化节点')

# # # Pretrained Model 2_8
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_model/num_hidden_layers_2/num_attention_heads_8/model.ckpt-1000000", '模型的初始化节点')

# # Pretrained Model 4_8
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_model/num_hidden_layers_4/num_attention_heads_8/model.ckpt-1000000", '模型的初始化节点')

# # Pretrained Model 6_8
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_model/num_hidden_layers_6/num_attention_heads_8/model.ckpt-1000000", '模型的初始化节点')

# # Pretrained Model 8_8
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_model/num_hidden_layers_8/num_attention_heads_8/model.ckpt-1000000", '模型的初始化节点')

# # Pretrained Model 10_8
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_model/num_hidden_layers_10/num_attention_heads_8/model.ckpt-1000000", '模型的初始化节点')

## Pretrained Model 12_8
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_model/num_hidden_layers_12/num_attention_heads_8/model.ckpt-1000000", '模型的初始化节点')

# Pretrained Model 12_12
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_model/num_hidden_layers_12/num_attention_heads_12/model.ckpt-1000000", '模型的初始化节点')


##################################################################

# # PPI_1_1 model
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_1_100.ckpt", '模型的初始化节点')

# PPI_1_2 model
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_2_100.ckpt", '模型的初始化节点')

# PPI_1_3 model
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_3_100.ckpt", '模型的初始化节点')

# # PPI_1_4 model
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_4_100.ckpt", '模型的初始化节点')

# PPI_1_6 model
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_6_100.ckpt", '模型的初始化节点')

# PPI_1_8 model
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_8_100.ckpt", '模型的初始化节点')

# PPI_1_12 model
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_12_100.ckpt", '模型的初始化节点')

# # PPI_2_8 model
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_2_8_100.ckpt", '模型的初始化节点')

# PPI_4_8 model
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_4_8_100.ckpt", '模型的初始化节点')

# # PPI_6_8 model
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_6_8_100.ckpt", '模型的初始化节点')

# # # PPI_12_8 model
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_12_8_100.ckpt", '模型的初始化节点')


# PPI_12_12 model 1 (train on 90% and eval on 10% for 100 epochs)
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI/model_PPI_100.ckpt", '模型的初始化节点')

# PPI_12_12 model 2 (train and eval on 90% (95% train & 5% eval) and test on 10% for 100 epochs)
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_12_12_100.ckpt", '模型的初始化节点')

# PPI_12_12 model 3 (train and eval on 90% (95% train & 5% eval) and test on 10% for 20 epochs)
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_20.ckpt", '模型的初始化节点')


############ PPI #####################
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_100.ckpt", '模型保存位置')
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_1_100_2.ckpt", '模型保存位置')

#tf.app.flags.DEFINE_string(
#    'save_path', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_2_100_2.ckpt", '模型保存位置')

#tf.app.flags.DEFINE_string(
#    'save_path', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_3_100_2.ckpt", '模型保存位置')

#tf.app.flags.DEFINE_string(
#    'save_path', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_4_100_2.ckpt", '模型保存位置')

#tf.app.flags.DEFINE_string(
#    'save_path', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_6_100_2.ckpt", '模型保存位置')

#tf.app.flags.DEFINE_string(
#    'save_path', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_8_100_2.ckpt", '模型保存位置')

# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_2_8_100.ckpt", '模型保存位置')

# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_4_8_100.ckpt", '模型保存位置')

# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_6_8_100.ckpt", '模型保存位置')

# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_8_8_100.ckpt", '模型保存位置')

# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_10_8_100.ckpt", '模型保存位置')

#tf.app.flags.DEFINE_string(
#    'save_path', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_12_100_2.ckpt", '模型保存位置')

# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_12_8_100.ckpt", '模型保存位置')

############ NbAg #####################

# Nb_Ag_1_1 (PreTrainBase)
tf.app.flags.DEFINE_string(
    'save_path', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_1_1/model_NbAg_11.ckpt", '模型保存位置')

# Nb_Ag_1_2 (PreTrainBase)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_1_2/model_NbAg_11.ckpt", '模型保存位置')

# Nb_Ag_1_3 (PreTrainBase)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_1_3/model_NbAg_11.ckpt", '模型保存位置')

# Nb_Ag_1_4 (PreTrainBase)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_1_4/model_NbAg_11.ckpt", '模型保存位置')

# # Nb_Ag_1_6 (PreTrainBase)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_1_6/model_NbAg_11.ckpt", '模型保存位置')

# # Nb_Ag_1_8 (PreTrainBase)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_1_8/model_NbAg_11.ckpt", '模型保存位置')

# # Nb_Ag_1_12 (PreTrainBase)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_1_12/model_NbAg_11.ckpt", '模型保存位置')

# # # Nb_Ag_2_8 (PreTrainBase)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_2_8/model_NbAg_10.ckpt", '模型保存位置')

# # Nb_Ag_4_8 (PreTrainBase)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_4_8/model_NbAg_1.ckpt", '模型保存位置')

# Nb_Ag_12_8 (PreTrainBase)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_12_8/model_NbAg_10.ckpt", '模型保存位置')



# Nb_Ag_12_12 (PreTrainBase)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/model_NbAg_100_PreTrainedBase_10.ckpt", '模型保存位置')

#######################
# # Nb_Ag_1_1 (PPIBase_1_1)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_1_1/model_NbAg_11.ckpt", '模型保存位置')

# Nb_Ag_1_2 (PPIBase_1_2)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_1_2/model_NbAg_11.ckpt", '模型保存位置')

# Nb_Ag_1_3 (PPIBase_1_3)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_1_3/model_NbAg_11.ckpt", '模型保存位置')

# # Nb_Ag_1_4 (PPIBase_1_4)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_1_4/model_NbAg_11.ckpt", '模型保存位置')

# # Nb_Ag_1_6 (PPIBase_1_6)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_1_6/model_NbAg_11.ckpt", '模型保存位置')

# # Nb_Ag_1_8 (PPIBase_1_8)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_1_8/model_NbAg_11.ckpt", '模型保存位置')

# # Nb_Ag_1_12 (PPIBase_1_12)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_1_12/model_NbAg_11.ckpt", '模型保存位置')

# # Nb_Ag_2_8 (PPIBase_2_8)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_2_8/model_NbAg_5.ckpt", '模型保存位置')

# # Nb_Ag_4_8 (PPIBase_4_8)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_4_8/model_NbAg_10.ckpt", '模型保存位置')

# # Nb_Ag_6_8 (PPIBase_6_8)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_6_8/model_NbAg_7.ckpt", '模型保存位置')

# # Nb_Ag_12_8 (PPIBase_12_8)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_12_8/model_NbAg_10.ckpt", '模型保存位置')


# Nb_Ag_12_12 (PPI_12_12_Base_Model_1)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_12_12/Model_1/model_NbAg_100_1_4.ckpt", '模型保存位置')

# Nb_Ag_12_12 (PPI_12_12_Base_Model_2)
# tf.app.flags.DEFINE_string(
#     'save_path', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_12_12/Model_2/model_NbAg_100_2_10.ckpt", '模型保存位置')

def count_trues(pre_labels, true_labels):
    shape = true_labels.shape
    zeros = np.zeros(shape=shape)
    ones = np.ones(shape=shape)
    pos_example_index = (true_labels == ones)
    neg_example_index = (true_labels == zeros)
    right_example_index = (pre_labels == true_labels)
    true_pos_examples = np.sum(np.logical_and(
        pos_example_index, right_example_index))
    true_neg_examples = np.sum(np.logical_and(
        neg_example_index, right_example_index))
    return np.sum(pos_example_index), np.sum(neg_example_index), true_pos_examples, true_neg_examples

def main():
    # 以下是输入的参数，当更换词典时，请修改文件 bert_config.json中的vocab_size的值
    do_eval = FLAGS.do_eval     # 是否在训练之后进行评估
    do_save_model = FLAGS.do_save_model    # 是否存储训练的模型
    data_name = FLAGS.data_name  # 选定数据名，用于导入对应的路径的数据

    #Dataset
    train_dict = {"PPI": 59956,
                  "NbAg": 1122,
                  }     # 记录了各个训练集的样本数量

    val_dict = {"PPI": 3156,
                  "NbAg": 60,
                  }     # 记录了各个训练集的样本数量

    tf.logging.set_verbosity(tf.logging.INFO)
    train_example_num = train_dict[data_name]   # 获取训练集样本数量
    val_example_num = val_dict[data_name]   # 获取训练集样本数量
    # test_example_num = test_dict[data_name]     # 获取测试集样本数量
    batch_size = FLAGS.batch_size  # batch的大小，如果gpu显存不够，可以考虑减小一下，尽量2的批次
    # 根据batch的大小计算出训练集的batch的数量
    train_batch_num = math.ceil(train_example_num / batch_size)
    val_batch_num = math.ceil(val_example_num / batch_size)       # 计算出测试集的batch的数量
    # test_batch_num = math.ceil(test_example_num / batch_size)       # 计算出测试集的batch的数量

    num_train_epochs = FLAGS.num_train_epochs   # 训练的次数，代表过几遍训练集
    warmup_proportion = FLAGS.warmup_proportion  # 预热的训练比例，保持不变即可
    learning_rate = FLAGS.learning_rate    # 学习率
    use_tpu = FLAGS.using_tpu         # 无视好了
    seq_length = FLAGS.seq_length        # 这边默认设定512，不需要更改
    data_root = FLAGS.data_root    # 根据实际调用的数据集来这里修改数据路径
    vocab_file = FLAGS.vocab_file             # 词典
    init_checkpoint = FLAGS.init_checkpoint  # 模型初始化检查点，记录了模型的权重
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config)    # 根据该文件可以配置模型的结构

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.75   # 这三行用来防止直接占用全GPU
    # 输入训练集，这个文件是采用ljy_tsv2record生成的
    # input_file = data_root + data_name + "_train.tf_record"
    input_file = data_root + "PPI_train.tf_record"
    #input_file = data_root + "NbAg_train.tf_record"
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    num_train_steps = int(
        train_example_num / batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)  # 配置训练次数
    input_ids = tf.placeholder(dtype=tf.int32, shape=(None, 512))
    input_mask = tf.placeholder(dtype=tf.int32, shape=(None, 512))
    segment_ids = tf.placeholder(dtype=tf.int32, shape=(None, 512))
    label_ids = tf.placeholder(dtype=tf.int32, shape=(None,))   # 留四个占位符，用以输入数据和标签
    is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
    is_training = True
    num_labels = 2  # 二分类，共两个标签
    use_one_hot_embeddings = False
    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)  # 该函数生成BERT模型的计算图，并返回计算图的总损失、各样本损失、两个输出
    tvars = tf.trainable_variables()

    if init_checkpoint:
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        # 到这里为止根据检查点初始化计算图变量，相当于加载模型
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    train_op = optimization.create_optimizer(
        total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)  # 初始化优化器，用于梯度下降使用
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }   # 生成输入数据所需的变量

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,))
        return d

    train_data = input_fn({"batch_size": batch_size})   # 生成训练集
    iterator = train_data.make_one_shot_iterator().get_next()   # 生成训练集数据迭代器，迭代器会在循环中输出数据
    if do_eval:
        # input_file = data_root + data_name + "_dev.tf_record"
        input_file = data_root + "PPI_val.tf_record"
        #input_file = data_root + "NbAg_val.tf_record"
        val_data = input_fn({"batch_size": batch_size})
        val_iterator = val_data.make_one_shot_iterator().get_next()  # 生成验证机迭代器

    val_accs = []
    val_sps = []
    val_sns = []
    if do_save_model:
        saver = tf.train.Saver()    # 生成存储节点
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)  # 初始化计算图

        start_time_all = time.time()

        for step in range(num_train_epochs):
            start_time = time.time()
            for _ in range(train_batch_num):
                examples = sess.run(iterator)  # 运行迭代器生成样本
                # print(examples)
                _, loss = \
                    sess.run([train_op, total_loss],
                             feed_dict={input_ids: examples["input_ids"],
                                        input_mask: examples["input_mask"],
                                        segment_ids: examples["segment_ids"],
                                        label_ids: examples["label_ids"]})  # 进行梯度下降，并取回loss值，喂入迭代器生成的数据
            print("step:", step, " loss:", round(loss, 4), end=" ")
            all_prob = []
            all_labels = []
            all_pre_labels = []
            if not do_eval:
                end_time = time.time()
                eta_time = (end_time - start_time) * \
                    (num_train_epochs - step - 1)
                print(" eta time:", eta_time, "s")
                continue
            for _ in range(val_batch_num):
                examples = sess.run(val_iterator)
                loss, prob = \
                    sess.run([total_loss, probabilities],
                             feed_dict={input_ids: examples["input_ids"],
                                        input_mask: examples["input_mask"],
                                        segment_ids: examples["segment_ids"],
                                        label_ids: examples["label_ids"]})
                all_prob.extend(prob[:, 1].tolist())
                all_labels.extend(examples["label_ids"].tolist())
                pre_labels = np.argmax(prob, axis=-1).tolist()
                all_pre_labels.extend(pre_labels)
            val_acc = accuracy_score(all_labels, all_pre_labels)
            val_accs.append(val_acc)
            val_auc = roc_auc_score(all_labels, all_prob)
            val_aupr = average_precision_score(all_labels, all_prob)
            val_mcc = matthews_corrcoef(all_labels, all_pre_labels)
            val_c_mat = confusion_matrix(all_labels, all_pre_labels)
            val_sn = val_c_mat[1, 1] / np.sum(val_c_mat[1, :])    # 预测正确的正样本
            val_sp = val_c_mat[0, 0] / np.sum(val_c_mat[0, :])    # 预测正确的负样本
            val_sps.append(val_sp)
            val_sns.append(val_sn)  # 计算各项性能指标
            end_time = time.time()
            eta_time = (end_time - start_time) * (num_train_epochs - step - 1)
            print("TN:", val_c_mat[0,0], "FP:", val_c_mat[0,1], "FN:", val_c_mat[1,0], "TP:", val_c_mat[1,1], "SE:", val_sn, " SP:", val_sp, " ACC:", val_acc, " MCC:", val_mcc, " auROC:", val_auc," aupr:",val_aupr ,  " eta time:", eta_time, "s")

        end_time_all = time.time()
        total_time = end_time_all - start_time_all
        print("Total training time: ", total_time, "s")
        if do_save_model:
            save_path = saver.save(
                sess, FLAGS.save_path)  # 存储模型

main()
