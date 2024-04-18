# 设置可以使用的GPU
# export CUDA_VISIBLE_DEVICES=1
python run_classifier_fatma.py \
  --data_name=NbAg \
  --data_root=./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/asTF_Record/ \
  --do_eval=True \
  --num_train_epochs=100 \
  --batch_size=64 \
  --bert_config=./bert_config_3.json \
  --vocab_file=./vocab/vocab_3kmer.txt \
  --init_checkpoint=./model/3kmer_model/num_hidden_layers_1/num_attention_heads_1/model.ckpt-1000000 \
  --save_path=./model/3kmer_Classifier_model_512/NbAg/PreTrainBase_1_1/model_NbAg_10.ckpt
