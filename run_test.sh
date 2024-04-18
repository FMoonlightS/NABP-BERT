# 设置可以使用的GPU
# export CUDA_VISIBLE_DEVICES=1
python run_classifier_fatma.py \
  --data_name=NbAg \
  --data_root=./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/asTF_Record/ \
  --do_test=True \
  --batch_size=64 \
  --bert_config=./bert_config_3.json \
  --vocab_file=./vocab/vocab_3kmer.txt \
  --init_checkpoint=./model/3kmer_Classifier_model_512/NbAg/NoPreTrain_1_1/model_NbAg_1.ckpt
