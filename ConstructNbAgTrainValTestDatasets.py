"""
@author: Fatma S. Ahmed
@brief: Construct Nb-Ag Train, Val, and Test Datasets.
"""

import pickle
from sklearn.model_selection import train_test_split
import random
random.seed(123)

with open('ConstructNbAgTrainValTestDatasets_4.txt','w') as f:
    f.write("loading the cleaned Nb-Ag pairs database from the local disk \n")
    print("loading the cleaned Nb-Ag pairs database from the local disk \n")

    with open('dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/Cleaned_Nb_Ag_pairs.pickle','rb') as binary_reader:
        Cleaned_Nb_Ag_pairs = pickle.load(binary_reader)
    print("len of Cleaned_Nb_Ag_pairs is : " + str(len(Cleaned_Nb_Ag_pairs))+ "\n")
    f.write("len of Cleaned_Nb_Ag_pairs is : " + str(len(Cleaned_Nb_Ag_pairs))+ "\n")

    # shuffle the data
    random.shuffle(Cleaned_Nb_Ag_pairs)

    # know the positive and negative pairs in trainCleaned_Nb_Ag_pairs and testCleaned_Nb_Ag_pairs
    pos = 0
    for pair in Cleaned_Nb_Ag_pairs:
        pos += pair[3]

    print("number of pos pair in Cleaned_Nb_Ag_pairs is : " + str(pos)+ "\n")
    print("number of neg pair in Cleaned_Nb_Ag_pairs is : " + str(len(Cleaned_Nb_Ag_pairs) - pos)+ "\n")
    f.write("number of pos pair in Cleaned_Nb_Ag_pairs is : " + str(pos)+ "\n")
    f.write("number of neg pair in Cleaned_Nb_Ag_pairs is : " + str(len(Cleaned_Nb_Ag_pairs) - pos)+ "\n")
    f.write("neg : pos ==> 1 : " + str((len(Cleaned_Nb_Ag_pairs) - pos)/pos) + "\n\n")

    # split the Cleaned_Nb_Ag_pairs into 90% for training and 10% for test
    print("split the Cleaned_Nb_Ag_pairs into 90% for training and 10% for test\n")
    f.write("split the Cleaned_Nb_Ag_pairs into 90% for training and 10% for test\n")
    trainCleaned_Nb_Ag_pairs, testCleaned_Nb_Ag_pairs = train_test_split(
        Cleaned_Nb_Ag_pairs, test_size=0.1, random_state=123)  # 90% train , 10% test

    print("len of trainCleaned_Nb_Ag_pairs is : " + str(len(trainCleaned_Nb_Ag_pairs))+ "\n")
    print("len of testCleaned_Nb_Ag_pairs is : " + str(len(testCleaned_Nb_Ag_pairs))+ "\n")
    f.write("len of trainCleaned_Nb_Ag_pairs is : " + str(len(trainCleaned_Nb_Ag_pairs))+ "\n")
    f.write("len of testCleaned_Nb_Ag_pairs is : " + str(len(testCleaned_Nb_Ag_pairs))+ "\n")

    # know the positive and negative pairs in trainCleaned_Nb_Ag_pairs and testCleaned_Nb_Ag_pairs
    pos = 0
    for pair in trainCleaned_Nb_Ag_pairs:
        pos += pair[3]

    print("number of pos pair in trainCleaned_Nb_Ag_pairs is : " + str(pos)+ "\n")
    print("number of neg pair in trainCleaned_Nb_Ag_pairs is : " + str(len(trainCleaned_Nb_Ag_pairs) - pos)+ "\n")
    f.write("number of pos pair in trainCleaned_Nb_Ag_pairs is : " + str(pos)+ "\n")
    f.write("number of neg pair in trainCleaned_Nb_Ag_pairs is : " + str(len(trainCleaned_Nb_Ag_pairs) - pos)+ "\n")
    f.write("neg : pos ==> 1 : " + str((len(trainCleaned_Nb_Ag_pairs) - pos)/pos) + "\n\n")


    pos = 0
    for pair in testCleaned_Nb_Ag_pairs:
        pos += pair[3]

    print("number of pos pair in testCleaned_Nb_Ag_pairs is : " + str(pos)+ "\n")
    print("number of neg pair in testCleaned_Nb_Ag_pairs is : " + str(len(testCleaned_Nb_Ag_pairs) - pos)+ "\n")
    f.write("number of pos pair in testCleaned_Nb_Ag_pairs is : " + str(pos)+ "\n")
    f.write("number of neg pair in testCleaned_Nb_Ag_pairs is : " + str(len(testCleaned_Nb_Ag_pairs) - pos)+ "\n")
    f.write("neg : pos ==> 1 : " + str((len(testCleaned_Nb_Ag_pairs) - pos)/pos) + "\n\n")

    print("store trainCleaned_Nb_Ag_pairs locally \n ")
    f.write("store trainCleaned_Nb_Ag_pairs locally \n ")
    with open('dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/trainCleaned_Nb_Ag_pairs.pickle','wb') as binary_writer:
        pickle.dump(trainCleaned_Nb_Ag_pairs,binary_writer)

    print("store testCleaned_Nb_Ag_pairs locally \n ")
    f.write("store testCleaned_Nb_Ag_pairs locally \n ")
    with open('dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/testCleaned_Nb_Ag_pairs.pickle','wb') as binary_writer:
        pickle.dump(testCleaned_Nb_Ag_pairs,binary_writer)

    ######################################################################

    # split the Cleaned_Nb_Ag_pairs into 95% for training and 5% for validation
    print("split the trainCleaned_Nb_Ag_pairs into 95% for training and 5% for validation\n")
    f.write("split the trainCleaned_Nb_Ag_pairs into 95% for training and 5% for validation\n")
    train2Cleaned_Nb_Ag_pairs, valCleaned_Nb_Ag_pairs = train_test_split(
        trainCleaned_Nb_Ag_pairs, test_size=0.05, random_state=123) # 95% train , 5% validation

    print("len of train2Cleaned_Nb_Ag_pairs is : " + str(len(train2Cleaned_Nb_Ag_pairs))+ "\n")
    print("len of valCleaned_Nb_Ag_pairs is : " + str(len(valCleaned_Nb_Ag_pairs))+ "\n")
    f.write("len of train2Cleaned_Nb_Ag_pairs is : " + str(len(train2Cleaned_Nb_Ag_pairs))+ "\n")
    f.write("len of valCleaned_Nb_Ag_pairs is : " + str(len(valCleaned_Nb_Ag_pairs))+ "\n")

    # know the positive and negative pairs in trainCleaned_Nb_Ag_pairs and testCleaned_Nb_Ag_pairs
    pos = 0
    for pair in train2Cleaned_Nb_Ag_pairs:
        pos += pair[3]

    print("number of pos pair in train2Cleaned_Nb_Ag_pairs is : " + str(pos)+ "\n")
    print("number of neg pair in train2Cleaned_Nb_Ag_pairs is : " + str(len(train2Cleaned_Nb_Ag_pairs) - pos)+ "\n")
    f.write("number of pos pair in train2Cleaned_Nb_Ag_pairs is : " + str(pos)+ "\n")
    f.write("number of neg pair in train2Cleaned_Nb_Ag_pairs is : " + str(len(train2Cleaned_Nb_Ag_pairs) - pos)+ "\n")
    f.write("neg : pos ==> 1 : " + str((len(train2Cleaned_Nb_Ag_pairs) - pos)/pos) + "\n\n")


    pos = 0
    for pair in valCleaned_Nb_Ag_pairs:
        pos += pair[3]

    print("number of pos pair in valCleaned_Nb_Ag_pairs is : " + str(pos)+ "\n")
    print("number of neg pair in valCleaned_Nb_Ag_pairs is : " + str(len(valCleaned_Nb_Ag_pairs) - pos)+ "\n")
    f.write("number of pos pair in valCleaned_Nb_Ag_pairs is : " + str(pos)+ "\n")
    f.write("number of neg pair in valCleaned_Nb_Ag_pairs is : " + str(len(valCleaned_Nb_Ag_pairs) - pos)+ "\n")
    f.write("neg : pos ==> 1 : " + str((len(valCleaned_Nb_Ag_pairs) - pos)/pos) + "\n\n")

    print("store train2Cleaned_Nb_Ag_pairs locally \n ")
    f.write("store train2Cleaned_Nb_Ag_pairs locally \n ")
    with open('dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/train2Cleaned_Nb_Ag_pairs.pickle','wb') as binary_writer:
        pickle.dump(train2Cleaned_Nb_Ag_pairs,binary_writer)

    print("store valCleaned_Nb_Ag_pairs locally \n ")
    f.write("store valCleaned_Nb_Ag_pairs locally \n ")
    with open('dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/valCleaned_Nb_Ag_pairs.pickle','wb') as binary_writer:
        pickle.dump(valCleaned_Nb_Ag_pairs,binary_writer)
