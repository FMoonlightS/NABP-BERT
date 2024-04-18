"""
@author: Fatma S. Ahmed
@brief: Create TSV files for Nb-Ag and PPI data.
"""

import pickle
import numpy as np
import random
from sklearn.model_selection import train_test_split

random.seed(123)

def sequence2tokens(seq, kmer):
    tokens = []
    for i in range(len(seq) - (kmer-1)):
        tokens.append(seq[i:i + kmer])
    return tokens

def PPI_create_tsv(firstProteins, secondProteins, mode, labels, kmer, firstProteinsTSV, secondProteinsTSV):
    f1 = open(firstProteinsTSV, "w")
    f2 = open(secondProteinsTSV, "w")

    for firstProtein, secondProtein, label in zip(firstProteins, secondProteins, labels):
        label_str = str(int(label[0]))
        protein = str(firstProtein[1]).lower()
        protein_tokens = ""
        tokens = sequence2tokens(protein, kmer)
        for token in tokens:
            protein_tokens = protein_tokens + token + " "
        res = mode + "\t" + label_str  + "\t\t" + protein_tokens
        f1.write(res + "\n")

        protein = str(secondProtein[1]).lower()
        protein_tokens = ""
        tokens = sequence2tokens(protein, kmer)
        for token in tokens:
            protein_tokens = protein_tokens + token + " "
        res = mode + "\t" + label_str  + "\t\t" + protein_tokens
        f2.write(res + "\n")
def NbAg_create_tsv(NbAg_pairs, mode, kmer, firstProteinsTSV, secondProteinsTSV):
    f1 = open(firstProteinsTSV, "w")
    f2 = open(secondProteinsTSV, "w")

    for  pair in NbAg_pairs:
        firstProtein = pair[0]
        secondProtein = pair[1]
        label = pair[3]

        label_str = str(label)
        protein = str(firstProtein).lower()
        protein_tokens = ""
        tokens = sequence2tokens(protein, kmer)
        for token in tokens:
            protein_tokens = protein_tokens + token + " "
        res = mode + "\t" + label_str  + "\t\t" + protein_tokens
        f1.write(res + "\n")

        protein = str(secondProtein).lower()
        protein_tokens = ""
        tokens = sequence2tokens(protein, kmer)
        for token in tokens:
            protein_tokens = protein_tokens + token + " "
        res = mode + "\t" + label_str  + "\t\t" + protein_tokens
        f2.write(res + "\n")

def PPI_createTrainValTestTSV():
    with open('createTSV_PPI.txt','w') as f:
        kmer = 3

        firstProteinsTSV  = "dataAfterPreProcessing/PPI_Dataset/asTSV/firstProteins_tr.tsv"
        secondProteinsTSV = "dataAfterPreProcessing/PPI_Dataset/asTSV/secondProteins_tr.tsv"

        f.write("load the train dataset from the local disk\n")
        with open("dataAfterPreProcessing/PPI_Dataset/firstProteinsListTrain.pickle", "rb") as output_:
            firstProteinsListTrain = pickle.load(output_)

        with open("dataAfterPreProcessing/PPI_Dataset/secondProteinsListTrain.pickle", "rb") as output_:
            secondProteinsListTrain = pickle.load(output_)

        with open("dataAfterPreProcessing/PPI_Dataset/thirdProteinsTensorTrain.pickle", "rb") as output_:
            thirdProteinsTensorTrain = pickle.load(output_)

        thirdProteinsListTrain = np.ndarray.tolist(thirdProteinsTensorTrain)

        Train_combine = list(zip(firstProteinsListTrain, secondProteinsListTrain, thirdProteinsListTrain))
        # print("hello")
        random.shuffle(Train_combine)

        # split the train dataset into 95% for train and 5% for validation

        Train2_combine, Val_combine = train_test_split(
            Train_combine, test_size=0.05, random_state=123) # 95% train , 5% validation

        # create tsv files for the training dataset
        firstProteinsListTrain, secondProteinsListTrain, thirdProteinsListTrain = zip(*Train2_combine)

        firstProteinsListTrain, secondProteinsListTrain, thirdProteinsListTrain = list(firstProteinsListTrain), list(secondProteinsListTrain), list(thirdProteinsListTrain)

        PPI_create_tsv(firstProteinsListTrain, secondProteinsListTrain, "train", thirdProteinsListTrain, kmer, firstProteinsTSV, secondProteinsTSV)

        # create tsv files for the validation dataset
        firstProteinsTSV  = "dataAfterPreProcessing/PPI_Dataset/asTSV/firstProteins_val.tsv"
        secondProteinsTSV = "dataAfterPreProcessing/PPI_Dataset/asTSV/secondProteins_val.tsv"

        firstProteinsListVal, secondProteinsListVal, thirdProteinsListVal = zip(*Val_combine)

        firstProteinsListVal, secondProteinsListVal, thirdProteinsListVal = list(firstProteinsListVal), list(secondProteinsListVal), list(thirdProteinsListVal)

        PPI_create_tsv(firstProteinsListVal, secondProteinsListVal, "val", thirdProteinsListVal, kmer, firstProteinsTSV, secondProteinsTSV)

    ###########################################################################
        firstProteinsTSV  = "dataAfterPreProcessing/PPI_Dataset/asTSV/firstProteins_te.tsv"
        secondProteinsTSV = "dataAfterPreProcessing/PPI_Dataset/asTSV/secondProteins_te.tsv"

        f.write("load the test dataset from the local disk\n")
        with open("dataAfterPreProcessing/PPI_Dataset/firstProteinsListTest.pickle", "rb") as output_:
            firstProteinsListTest = pickle.load(output_)

        with open("dataAfterPreProcessing/PPI_Dataset/secondProteinsListTest.pickle", "rb") as output_:
            secondProteinsListTest= pickle.load(output_)

        with open("dataAfterPreProcessing/PPI_Dataset/thirdProteinsTensorTest.pickle", "rb") as output_:
            thirdProteinsTensorTest = pickle.load(output_)

        thirdProteinsListTest = np.ndarray.tolist(thirdProteinsTensorTest)

        Test_combine = list(zip(firstProteinsListTest, secondProteinsListTest, thirdProteinsListTest))
        # print("hello")
        random.shuffle(Test_combine)

        firstProteinsListTest, secondProteinsListTest, thirdProteinsListTest = zip(*Test_combine)

        firstProteinsListTest, secondProteinsListTest, thirdProteinsListTest = list(firstProteinsListTest), list(secondProteinsListTest), list(thirdProteinsListTest)

        PPI_create_tsv(firstProteinsListTest, secondProteinsListTest, "test", thirdProteinsListTest, kmer, firstProteinsTSV, secondProteinsTSV)

def NbAg_createTrainValTestTSV():
    with open('createTSV_NbAg.txt', 'w') as f:
        kmer = 3

        # create tsv files for the train dataset
        firstProteinsTSV = "dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/asTSV/firstProteins_tr.tsv"
        secondProteinsTSV = "dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/asTSV/secondProteins_tr.tsv"

        f.write("load train2Cleaned_Nb_Ag_pairs from the local disk \n")
        with open('dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/train2Cleaned_Nb_Ag_pairs.pickle',
                  'rb') as binary_reader:
            train2Cleaned_Nb_Ag_pairs = pickle.load(binary_reader)

        NbAg_create_tsv(train2Cleaned_Nb_Ag_pairs, "train", kmer,
                   firstProteinsTSV, secondProteinsTSV)

        # create tsv files for the validation dataset
        firstProteinsTSV = "dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/asTSV/firstProteins_val.tsv"
        secondProteinsTSV = "dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/asTSV/secondProteins_val.tsv"

        f.write("load valCleaned_Nb_Ag_pairs from the local disk \n")
        with open('dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/valCleaned_Nb_Ag_pairs.pickle',
                  'rb') as binary_reader:
            valCleaned_Nb_Ag_pairs = pickle.load(binary_reader)

        NbAg_create_tsv(valCleaned_Nb_Ag_pairs, "val", kmer,
                   firstProteinsTSV, secondProteinsTSV)


        # create tsv files for the test dataset
        firstProteinsTSV = "dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/asTSV/firstProteins_te.tsv"
        secondProteinsTSV = "dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/asTSV/secondProteins_te.tsv"

        f.write("load testCleaned_Nb_Ag_pairs from the local disk \n")
        with open('dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/testCleaned_Nb_Ag_pairs.pickle',
                  'rb') as binary_reader:
            testCleaned_Nb_Ag_pairs = pickle.load(binary_reader)

        NbAg_create_tsv(testCleaned_Nb_Ag_pairs, "test", kmer,
                   firstProteinsTSV, secondProteinsTSV)


# PPI_createTrainValTestTSV()
NbAg_createTrainValTestTSV()
print("hello")