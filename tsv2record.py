"""
@author: Fatma S. Ahmed
@brief: convert tsv files to  TFrecord file.
"""

# coding:utf-8

from run_classifier import ColaProcessor
import tokenization
from run_classifier import file_based_convert_examples_to_features
import random

def create_tfrecord(kmer):
    # tsv_root = "dataAfterPreProcessing/PPI_Dataset/asTSV_1/"
    # tfrecord_root = "dataAfterPreProcessing/PPI_Dataset/asTF_Record_1/"

    # tsv_root = "dataAfterPreProcessing/PPI_Dataset/asTSV_2/"
    # tfrecord_root = "dataAfterPreProcessing/PPI_Dataset/asTF_Record_2/"

    tsv_root = "dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/asTSV/"
    tfrecord_root = "dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/asTF_Record/"

    # tsv_root = "Dataset/asTSV/" + str(kmer) + "kmer_tsv_data/" + cell_line
    # tfrecord_root = "Dataset/asTF_Record/" + str(kmer) + "kmer_tfrecord/" + cell_line

    vocab_file = "vocab/vocab_" + str(kmer) + "kmer.txt"
    processor = ColaProcessor()
    label_list = processor.get_labels()

    # create tfRecord file for the test dataset
    examples = processor.fatma_get_test_examples(tsv_root)
    # test_file = tfrecord_root + "PPI_test.tf_record"
    test_file = tfrecord_root + "NbAg_test.tf_record"
    tokenizer = tokenization.FullTokenizer(
          vocab_file=vocab_file, do_lower_case=True)
    file_based_convert_examples_to_features(
            examples, label_list, 512, tokenizer, test_file)

    # create tfRecord file for the validation dataset
    examples = processor.fatma_get_val_examples(tsv_root)
    # val_file = tfrecord_root + "PPI_val.tf_record"
    val_file = tfrecord_root + "NbAg_val.tf_record"
    tokenizer = tokenization.FullTokenizer(
          vocab_file=vocab_file, do_lower_case=True)
    file_based_convert_examples_to_features(
            examples, label_list, 512, tokenizer, val_file)

    # create tfRecord file for the training dataset
    examples = processor.fatma_get_train_examples(tsv_root)
    # train_file = tfrecord_root + "PPI_train.tf_record"
    train_file = tfrecord_root + "NbAg_train.tf_record"
    tokenizer = tokenization.FullTokenizer(
          vocab_file=vocab_file, do_lower_case=True)
    file_based_convert_examples_to_features(
            examples, label_list, 512, tokenizer, train_file)



kmer = 3
create_tfrecord(kmer)