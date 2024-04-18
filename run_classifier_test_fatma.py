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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('do_test', True, '是否在训练后评估')
# # # NbAg_NoPreTrain_1_1 Model
tf.app.flags.DEFINE_string(
    'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/NoPreTrain_1_1/model_NbAg_1.ckpt", '模型的初始化节点')

# tf.app.flags.DEFINE_string('data_name', 'PPI', '导入的数据名')
tf.app.flags.DEFINE_string('data_name', 'NbAg', '导入的数据名')

tf.app.flags.DEFINE_integer('batch_size', 64, 'batch大小')
tf.app.flags.DEFINE_integer('num_train_epochs', 1, '训练的轮次')
tf.app.flags.DEFINE_float('warmup_proportion', 0.1, '预热的训练比例')
tf.app.flags.DEFINE_float('learning_rate', 2e-5, '学习率')
tf.app.flags.DEFINE_boolean('using_tpu', False, '是否使用TPU')
tf.app.flags.DEFINE_float('seq_length', 512, '序列长度')

# tf.app.flags.DEFINE_string(
#         'data_root', './dataAfterPreProcessing/PPI_Dataset/PreProcessing_2/asTF_Record_2/', '使用数据集的根目录')

tf.app.flags.DEFINE_string(
        'data_root', './dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/PreProcessing_2/asTF_Record/', '使用数据集的根目录')

tf.app.flags.DEFINE_string('bert_config', "./bert_config_3.json", 'bert配置')

tf.app.flags.DEFINE_string('vocab_file', './vocab/vocab_3kmer.txt', '词典目录')

###########  PPI #################################
# PPI_12_12_100 Model 1
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI/model_PPI_100.ckpt", '模型的初始化节点')

# PPI_12_12_100 Model 2
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_100.ckpt", '模型的初始化节点')

# PPI_12_12_20 Model 3
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_20.ckpt", '模型的初始化节点')

# PPI_1_1_100
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_1_100.ckpt", '模型的初始化节点')

# PPI_1_2_100
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_2_100.ckpt", '模型的初始化节点')

# # PPI_1_3_100
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_3_100.ckpt", '模型的初始化节点')

# # PPI_1_4_100
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_4_100.ckpt", '模型的初始化节点')

# # PPI_1_6_100
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_6_100.ckpt", '模型的初始化节点')

# PPI_1_8_100
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_8_100.ckpt", '模型的初始化节点')

# PPI_1_12_100
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/PPI_2/model_PPI_1_12_100.ckpt", '模型的初始化节点')

################  NbAg   ###########################
## NbAg_12_12 Model 1 (PPI_12_12 Base Model 1)
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_12_12/Model_1/model_NbAg_100_1_4.ckpt", '模型的初始化节点')

# # NbAg_12_12 Model 2 (PPI_12_12 Base Model 2)
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_12_12/Model_2/model_NbAg_100_2_10.ckpt", '模型的初始化节点')

# NbAg_12_12 Model 3 (PPI_12_12 Base Model 3)
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/model_NbAg_100_3.ckpt", '模型的初始化节点')

# NbAg_1_1 Model  (PPI_1_1 Base )
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_1_1/model_NbAg_10.ckpt", '模型的初始化节点')

# # NbAg_1_2 Model  (PPI_1_2 Base )
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_1_2/model_NbAg_10.ckpt", '模型的初始化节点')

# # NbAg_1_3 Model  (PPI_1_3 Base )
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_1_3/model_NbAg_10.ckpt", '模型的初始化节点')

# # NbAg_1_4 Model  (PPI_1_4 Base )
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_1_4/model_NbAg_10.ckpt", '模型的初始化节点')

# # # NbAg_1_6 Model  (PPI_1_6 Base )
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_1_6/model_NbAg_10.ckpt", '模型的初始化节点')


# # # NbAg_1_12 Model  (PPI_1_12 Base )
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PPIBase_1_12/model_NbAg_10.ckpt", '模型的初始化节点')

## NbAg_12_12 Model PretrainedBase_12_12
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_12_12/model_NbAg_100_PreTrainedBase_10.ckpt", '模型的初始化节点')

# # NbAg_1_1 Model PretrainedBase_1_1
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_1_1/model_NbAg_10.ckpt", '模型的初始化节点')

# # NbAg_1_2 Model PretrainedBase_1_2
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_1_2/model_NbAg_10.ckpt", '模型的初始化节点')

# # # NbAg_1_3 Model PretrainedBase_1_3
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_1_3/model_NbAg_10.ckpt", '模型的初始化节点')

# # NbAg_1_4 Model PretrainedBase_1_4
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_1_4/model_NbAg_10.ckpt", '模型的初始化节点')

# # NbAg_1_6 Model PretrainedBase_1_6
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_1_6/model_NbAg_10.ckpt", '模型的初始化节点')

# # NbAg_1_8 Model PretrainedBase_1_8
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_1_8/model_NbAg_10.ckpt", '模型的初始化节点')

# # NbAg_1_8 Model PretrainedBase_1_12
# tf.app.flags.DEFINE_string(
#     'init_checkpoint', "./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_1_12/model_NbAg_10.ckpt", '模型的初始化节点')

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
    do_test = FLAGS.do_test     # 是否在训练之后进行评估
    # do_save_model = FLAGS.do_save_model    # 是否存储训练的模型
    data_name = FLAGS.data_name  # 选定数据名，用于导入对应的路径的数据
    #Dataset
    test_dict = {"PPI": 562,
                 "NbAg": 132,
                 }  # 记录了各个测试集的样本数量


    tf.logging.set_verbosity(tf.logging.INFO)
    test_example_num = test_dict[data_name]     # 获取测试集样本数量
    batch_size = FLAGS.batch_size  # batch的大小，如果gpu显存不够，可以考虑减小一下，尽量2的批次
    # 根据batch的大小计算出训练集的batch的数量
    test_batch_num = math.ceil(
        test_example_num / batch_size)       # 计算出测试集的batch的数量
    num_train_epochs = FLAGS.num_train_epochs   # 训练的次数，代表过几遍训练集
    seq_length = FLAGS.seq_length        # 这边默认设定128，不需要更改
    data_root = FLAGS.data_root    # 根据实际调用的数据集来这里修改数据路径
    init_checkpoint = FLAGS.init_checkpoint  # 模型初始化检查点，记录了模型的权重
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config)    # 根据该文件可以配置模型的结构

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.75   # 这三行用来防止直接占用全GPU
    # 输入训练集，这个文件是采用ljy_tsv2record生成的
    # input_file = data_root + "PPI_test.tf_record"
    input_file = data_root + "NbAg_test.tf_record"
    input_ids = tf.placeholder(dtype=tf.int32, shape=(None, 512))
    input_mask = tf.placeholder(dtype=tf.int32, shape=(None, 512))
    segment_ids = tf.placeholder(dtype=tf.int32, shape=(None, 512))
    label_ids = tf.placeholder(dtype=tf.int32, shape=(None,))   # 留四个占位符，用以输入数据和标签
    is_training = False
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

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }   # 生成输入数据所需的变量
    drop_remainder = False

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

    if do_test:
        # input_file = data_root + "PPI_test.tf_record"
        input_file = data_root + "NbAg_test.tf_record"
        test_data = input_fn({"batch_size": batch_size})
        test_iterator = test_data.make_one_shot_iterator().get_next()  # 生成验证机迭代器

    test_val_accs = []
    test_sps = []
    test_sns = []
    # if do_save_model:
    #     saver = tf.train.Saver()    # 生成存储节点

    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)  # 初始化计算图

        for step in range(num_train_epochs):
            start_time = time.time()
            all_prob = []
            all_labels = []
            all_pre_labels = []
            if not do_test:
                end_time = time.time()
                eta_time = (end_time - start_time) * \
                    (num_train_epochs - step - 1)
                print(" eta time:", eta_time, "s")
                continue
            for _ in range(test_batch_num):
                examples = sess.run(test_iterator)
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
            test_acc = accuracy_score(all_labels, all_pre_labels)
            test_val_accs.append(test_acc)
            test_auc = roc_auc_score(all_labels, all_prob)
            test_aupr = average_precision_score(all_labels, all_prob)
            test_mcc = matthews_corrcoef(all_labels, all_pre_labels)
            test_c_mat = confusion_matrix(all_labels, all_pre_labels)
            test_sn = test_c_mat[1, 1] / np.sum(test_c_mat[1, :])    # 预测正确的正样本
            test_sp = test_c_mat[0, 0] / np.sum(test_c_mat[0, :])    # 预测正确的负样本
            test_sps.append(test_sp)
            test_sns.append(test_sn)  # 计算各项性能指标
            end_time = time.time()
            eta_time = (end_time - start_time) * (num_train_epochs - step - 1)
            test_time = end_time - start_time
            print("TN:", test_c_mat[0,0], "FP:", test_c_mat[0,1], "FN:", test_c_mat[1,0], "TP:", test_c_mat[1,1], "SE:", test_sn, " SP:", test_sp, " ACC:", test_acc, " MCC:", test_mcc, " auROC:", test_auc," aupr:",test_aupr ,  " eta time:", eta_time, "s")
            print("test time : ", test_time, "s")


main()
