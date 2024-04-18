python create_pretraining_data.py \
	--input_file=./dataAfterPreProcessing/PreTrainDataset/PretrainData_Final.txt \
	--output_file=./dataAfterPreProcessing/PreTrainDataset/PretrainData_Final.tfrecord \
	--vocab_file=./vocab/vocab_3kmer.txt \
	--do_lower_case=True \
	--max_seq_length=512 \
	--max_predictions_per_seq=20 \
	--masked_lm_prob=0.15 \
	--random_seed=12345 \
	--dupe_factor=5
