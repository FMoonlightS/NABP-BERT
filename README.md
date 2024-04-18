# NABP-BERT

# Introduction
NABP-BERT is a model for Nanobody-Antigen Binding Prediction (NABP)  based on BERT.
This study presents PROT-BERT, a specialized self-supervised pre-training model designed to learn detailed representations of general protein sequences. This model can serve as a pre-trained model for subsequent protein-related tasks.  Furthermore, we introduce PPI-PROT-BERT, a supervised model trained to predict the interaction between a pair of protein sequences. PPI-PROT-BERT utilizes PROT-BERT as its pre-trained model and serves as a foundation for downstream protein-protein interaction tasks. Based on these models, we introduce  NABP-BERT, a novel supervised model that predicts the binding between a pair of nanobody-antigen sequences. NABP-BERT is available in two variants: NABP-PROT-BERT and NABP-PPI-PROT-BERT, which utilize PROT-BERT and PPI-PROT-BERT as pre-trained models, respectively.
To the best of our knowledge, NABP-BERT  represents the first attempt to predict Nb-Ag binding using the BERT model. Our main contributions involve the following: (i) presenting PROT-BERT, a self-supervised model for protein tasks; (ii) demonstrating that the K-mer approach is better for encoding protein sequences and combining pairs of sequences for predicting nanobody-antigen interaction; (iii) introducing PPI-PROT-BERT, a supervised model for protein-protein interaction prediction tasks; (iv) proposing NABP-PROT-BERT and NABP-PPI-PROT-BERT,  two supervised models to predict the binding between nanobody-antigen pairs; (iiv) NABP-BERT models outperform other state-of-the-art methods.
# How to Start
You should first clone the project by command
>git clone https://github.com/FMoonlightS/NABP-BERT.git

Then you need to download the models from this link:
>https://drive.google.com/file/d/1Sr3VMZ96z6duEvAaS6Fb4XtNBqaFX219/view?usp=sharing
 
Then you need to download the datasets from these links:
- PPI data: http://hint.yulab.org/download/ (note: download the Binary files)
- Uniport data: https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/ (note: download uniprot_sprot.fasta.gz and uniprot_trembl.fasta.gz)

Then you should uzip these zips and put them on the data folder of the project.

# Requirments 
We recommend you to build a python virtual environment with Anaconda. Also, please make sure you have at least one NVIDIA GPU with Linux x86_64 Driver Version = 430.26 (compatible with CUDA 10.2), then install the following packages in the virtual envirnoment

python -->                    3.7.6

tensorflow -->                1.15.0

tensorflow-gpu -->            2.1.0

six -->                       1.13.0

numpy -->                     1.17.4

scikit-learn -->              0.22

# Pre-processing the data
First you should prepare your data that will be used for pre-training and fine-tuning by running all the following commands in order

A) Nb-Ag data
> execute the code in this notebook "Nb_Ag_Sequence_Processing.ipynb" using google colab

> python CleanNbAgSeqs.py 
  
> python ConstructNbAgTrainValTestDatasets.py

B) PPI data
> python createDatabase.py

> python get_unmappedKeysInSwissPortDatabase.py

> python ExtractingProteinSeqsFromTrembleDatabase.py

> python CleanHINTdatabase.py

> python PrepearPPIDatasetSequences.py

> python Remove_Homology.py

> python ConstructPPINegativeSamples.py
 
C) Create TSV and TF_Record files for Nb-Ag and PPI data
> python createTSV.py

> python tsv2record.py

D) Pre-train data
> python Prepear_PreTrain_Dataset.py

# Pre-training
 
You should create data for pre-train by the command
>sh create_data.sh
 
You should ensure the content of file pre_train.sh
- input_file is your input data for pre-training whose format is tf_record.  
- output_dir is the dir of your output model.
- bert_config_file defines the structure of the model.
- train_batch_size should be change more little if your computer don't support so big batch size.
- You can change the num_train_steps by yourself.

After ensuring the content, then you can pre-train your model by the command:
>sh pre_train.sh

# Fine-Tuning & Evaluation & Save Model
When you ready to fine-tune the model, you should open this file first:
file run_fine_tune.sh, then you should change the parameters according to your needs.
- do_eval and do_save are used to indicate if you want to evaluate the model or save the final model.  
- If the do_save is True then you should specify the location to save the final model.
- init_chechpoint is the model which is used to train.

After ensuring the content, then you can fine-tune your model by this command:
> sh run_fine_tune.sh

# test
After fine-tuning the model, to test the model you should open this file "run_test.sh", and modify its content according to your need then run this command
> sh run_test.sh
