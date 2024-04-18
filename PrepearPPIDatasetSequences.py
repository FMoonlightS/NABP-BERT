"""
@author: Fatma S. Ahmed
@brief: Prepear PPI Dataset Sequences.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# load the modules':

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
# import tensorflow as tf
import random
from random import sample
import pandas as pd
# from utils import shuffleArray, splitSequenceintoKmers
import os
random.seed(123)
with open('PrepearPPIDatasetSequences_PreProcessing_2_blast.txt','w') as f:
    # load the binnarized dataset:
    with open("dataAfterPreProcessing/TrembleDataBase/curratedPPIDataBaseFromTremble.pickle", "rb") as input_:
        database_Tremble = pickle.load(input_)
    print("size of curratedPPIDataBaseFromTremble database : " + str(len(database_Tremble)) + "\n")
    f.write("size of curratedPPIDataBaseFromTremble database : " + str(len(database_Tremble)) + "\n")

    with open("dataAfterPreProcessing/SwissDataBase/SwissDataBase.pickle","rb") as input_:
        database_Swiss=pickle.load(input_)
    print("size of Swissport database : " + str(len(database_Swiss)) + "\n")
    f.write("size of Swissport database : " + str(len(database_Swiss)) + "\n")

    with open('dataAfterPreProcessing/PPI_Dataset/unique_protein_set.pickle','rb') as input_:
        PPI_unique_protein_set=pickle.load(input_)
    print("size of unique_protein_set database : " + str(len(PPI_unique_protein_set)) + "\n")
    f.write("size of unique_protein_set database : " + str(len(PPI_unique_protein_set)) + "\n")

   # remove Cleaned_Nb_Ag_pairs_unique_UniProtIDs from PPI_unique_protein_set
    with open('dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/Cleaned_Nb_Ag_pairs_unique_UniProtIDs.pickle','rb') as input_:
        Cleaned_Nb_Ag_pairs_unique_UniProtIDs=pickle.load(input_)

    PPI_unique_UniProtID_after_Remove_Nb_Ag_IDs = set()
    for UniPortID in PPI_unique_protein_set:
        if UniPortID in Cleaned_Nb_Ag_pairs_unique_UniProtIDs:
            continue
        else:
            PPI_unique_UniProtID_after_Remove_Nb_Ag_IDs.add(UniPortID)

    print("store the PPI_unique_UniProtID_after_Remove_Nb_Ag_IDs to the disk \n")
    f.write("store the PPI_unique_UniProtID_after_Remove_Nb_Ag_IDs to the disk \n")
    with open("dataAfterPreProcessing/PPI_Dataset/PPI_unique_UniProtID_after_Remove_Nb_Ag_IDs.pickle","wb") as output_:
        pickle.dump(PPI_unique_UniProtID_after_Remove_Nb_Ag_IDs,output_)

    # Checking the validitiy of the keys:
    print("Checking the validitiy of the keys:\n")
    f.write("Checking the validitiy of the keys:\n")
    curratedKeys=set()
    unmappedKeys=set()
    TrembleKeys=database_Tremble.keys()
    SwissKeys=database_Swiss.keys()
    for key in PPI_unique_UniProtID_after_Remove_Nb_Ag_IDs:
        if key in SwissKeys:
            curratedKeys.add(key)
        else:
            if key in TrembleKeys:
                curratedKeys.add(key)
            else:
                unmappedKeys.add(key)

    print("Number of PPI unique keys before Remove Nb-Ag unique IDs: "+str(len(PPI_unique_protein_set)) + "\n")
    print("Number of PPI unique keys after Remove Nb-Ag unique IDs: "+str(len(PPI_unique_UniProtID_after_Remove_Nb_Ag_IDs)) + "\n")
    print("Number of mapped keys: "+str(len(curratedKeys))+ "\n")
    print("Number of unmapped keys: "+str(len(unmappedKeys))+ "\n")
    print("Success Percentage: "+str(
            (len(curratedKeys)/len(PPI_unique_UniProtID_after_Remove_Nb_Ag_IDs))*100)+"%" + "\n")

    f.write("Number of PPI unique keys before Remove Nb-Ag unique IDs: "+str(len(PPI_unique_protein_set)) + "\n")
    f.write("Number of PPI unique keys after Remove Nb-Ag unique IDs: "+str(len(PPI_unique_UniProtID_after_Remove_Nb_Ag_IDs)) + "\n")
    f.write("Number of mapped keys: "+str(len(curratedKeys))+ "\n")
    f.write("Number of unmapped keys: "+str(len(unmappedKeys))+ "\n")
    f.write("Success Percentage: "+str(
            (len(curratedKeys)/len(PPI_unique_UniProtID_after_Remove_Nb_Ag_IDs))*100)+"%" + "\n")

    ## Extracting the protein sequences:
    curratedPPIDataBase=dict()
    for key in curratedKeys:
        try:
            curratedPPIDataBase[key]=database_Swiss[key]
        except:
            try:
                curratedPPIDataBase[key]=database_Tremble[key]
            except:
                print("I can not Extract the sequence of " + key + " !" + "\n")
                f.write("I can not Extract the sequence of " + key + " !" + "\n")
                pass
    print("Sequences have been extracted \n")
    print('numOfSeq : ' + str(len(curratedPPIDataBase)) + "\n")
    print("Saving the generated database locally \n")
    f.write("Sequences have been extracted \n")
    f.write('numOfSeq : ' + str(len(curratedPPIDataBase)) + "\n")
    f.write("Saving the generated database locally \n")

    with open("dataAfterPreProcessing/PPI_Dataset/curratedDataBase.pickle","wb") as output_:
        pickle.dump(curratedPPIDataBase,output_)

    # print('hello')
    #
    #
    # Cleaning the database by length:
    sequences_length=[]
    for _, seq in curratedPPIDataBase.items():
        sequences_length.append(len(seq))
    print("max seq len in curratedPPIDataBase : " + str(max(sequences_length)) + "\n")
    print("min seq len in curratedPPIDataBase : " + str(min(sequences_length)) + "\n")

    f.write("max seq len in curratedPPIDataBase : " + str(max(sequences_length)) + "\n")
    f.write("min seq len in curratedPPIDataBase : " + str(min(sequences_length)) + "\n")

    # load the database:
    with open("dataAfterPreProcessing/PPI_Dataset/set_pairs.pickle","rb") as input_:
        interactingPair=pickle.load(input_)

    ## Clean The PPI DataBase based on making the len of comining tokens from a pair of sequences to be <= 509 (max len of BERT model 509+3=512)
    cleanedByLengthInteractingPair=[]
    cleanedByLengthInteractingPairSeqs=[]
    # firstProteinPos, secondProteinPos = [], []
    for pair in interactingPair:
        if((pair[0] in curratedKeys) and (pair[1] in curratedKeys)):
            if ((len(curratedPPIDataBase[pair[0]])-3+1) + (len(curratedPPIDataBase[pair[1]])-3+1) <=509):
                cleanedByLengthInteractingPair.append(pair)
                cleanedByLengthInteractingPairSeqs.append((pair[0], curratedPPIDataBase[pair[0]], pair[1], curratedPPIDataBase[pair[1]], 1))
                # firstProteinPos.append(curratedPPIDataBase[pair[0]])
                # secondProteinPos.append(curratedPPIDataBase[pair[1]])

    # assert len(firstProteinPos) == len(secondProteinPos)
    # labelsPositive = np.ones((len(firstProteinPos))).reshape(-1, 1)

    print("Number of Interacting pairs before cleaning by length: "+str(len(interactingPair)) + "\n")
    print("Number of Interacting pairs after cleaning by length from keys: "+str(len(cleanedByLengthInteractingPair)) + "\n")
    print("Number of Interacting pairs after cleaning by length from seqs: "+str(len(cleanedByLengthInteractingPairSeqs)) + "\n")
    print("Cleaning the PPI DataBase contains "+str(
            (len(cleanedByLengthInteractingPair)/len(interactingPair)*100)
            )+"% of the uncleaned database\n")
    print("Reduction Ration: "+str(1-(len(cleanedByLengthInteractingPair)/len(interactingPair)))+"\n")

    f.write("Number of Interacting pairs before by length cleaning: "+str(len(interactingPair)) + "\n")
    f.write("Number of Interacting pairs after cleaning by length from keys: "+str(len(cleanedByLengthInteractingPair)) + "\n")
    f.write("Number of Interacting pairs after cleaning by length from seqs: "+str(len(cleanedByLengthInteractingPairSeqs)) + "\n")
    f.write("Cleaning the PPI DataBase contains "+str(
            (len(cleanedByLengthInteractingPair)/len(interactingPair)*100)
            )+"% of the uncleaned database\n")
    f.write("Reduction Ration: "+str(1-(len(cleanedByLengthInteractingPair)/len(interactingPair)))+"\n")

    print("writing the cleaning By Length database to the disk \n")
    f.write("writing the cleaning ByLength database to the disk \n")
    with open("dataAfterPreProcessing/PPI_Dataset/CleanedByLengthInteractionPaired.pickle","wb") as output_:
        pickle.dump(cleanedByLengthInteractingPair,output_)
    with open("dataAfterPreProcessing/PPI_Dataset/CleanedByLengthInteractionPairedSeqs.pickle","wb") as output_:
        pickle.dump(cleanedByLengthInteractingPairSeqs,output_)

    ## Split the unique identifer set and sequences after cleaning By Length into training and test datasets:
    trainInteractingPair_Seqs, testInteractingPair_Seqs = train_test_split(
        cleanedByLengthInteractingPairSeqs, test_size=0.1)

    print("number of trainInteractingPair_Seqs (list) : " + str(len(trainInteractingPair_Seqs)) + "\n")
    print("number of testInteractingPair_Seqs (list) : " + str(len(testInteractingPair_Seqs)) + "\n")
    f.write("number of trainInteractingPair_Seqs (list) : " + str(len(trainInteractingPair_Seqs)) + "\n")
    f.write("number of testInteractingPair_Seqs (list) : " + str(len(testInteractingPair_Seqs)) + "\n")

    ## Writing the trainInteractingPair_Seqs and testInteractingPair_Seqs to the disk:
    with open("dataAfterPreProcessing/PPI_Dataset/trainInteractingPair_Seqs_list.pickle", "wb") as output_:
        pickle.dump(trainInteractingPair_Seqs, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/testInteractingPair_Seqs_list.pickle", "wb") as output_:
        pickle.dump(testInteractingPair_Seqs, output_)

    # write the training dataset (list) as a FASTA file:
    Train_ProtA = open("dataAfterPreProcessing/PPI_Dataset/trainingSequencesDataSetProteinA_fromList.fasta", "w")
    Train_ProtB = open("dataAfterPreProcessing/PPI_Dataset/trainingSequencesDataSetProteinB_fromList.fasta", "w")
    for pair in trainInteractingPair_Seqs:
        Train_ProtA.write(">" + pair[0] + " \n")
        Train_ProtA.write(pair[1] + " \n")
        Train_ProtB.write(">" + pair[2] + " \n")
        Train_ProtB.write(pair[3] + " \n")

    # write the test dataset (list) as a FASTA file:
    Test_ProtA = open("dataAfterPreProcessing/PPI_Dataset/testSequencesDataSetProteinA_fromList.fasta", "w")
    Test_ProtB = open("dataAfterPreProcessing/PPI_Dataset/testSequencesDataSetProteinB_fromList.fasta", "w")
    for pair in testInteractingPair_Seqs:
        Test_ProtA.write(">" + pair[0] + " \n")
        Test_ProtA.write(pair[1] + " \n")
        Test_ProtB.write(">" + pair[2] + " \n")
        Test_ProtB.write(pair[3] + " \n")
    print("*********************************************************************")

    trainInteractingPair_Seqs = set(trainInteractingPair_Seqs)
    testInteractingPair_Seqs  = set(testInteractingPair_Seqs)

    print("number of trainInteractingPair_Seqs (set) : " + str(len(trainInteractingPair_Seqs)) + "\n")
    print("number of testInteractingPair_Seqs (set) : " + str(len(testInteractingPair_Seqs)) + "\n")
    f.write("number of trainInteractingPair_Seqs (set) : " + str(len(trainInteractingPair_Seqs)) + "\n")
    f.write("number of testInteractingPair_Seqs (set) : " + str(len(testInteractingPair_Seqs)) + "\n")

    ## Writing the trainInteractingPair_Seqs and testInteractingPair_Seqs to the disk:
    with open("dataAfterPreProcessing/PPI_Dataset/trainInteractingPair_Seqs_set.pickle", "wb") as output_:
        pickle.dump(trainInteractingPair_Seqs, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/testInteractingPair_Seqs_set.pickle", "wb") as output_:
        pickle.dump(testInteractingPair_Seqs, output_)

    ## Remove by Homology:
    # note: we have ProtA and ProtB seqs in the training and test datasets
    # write the training dataset (set) as a FASTA file:
    Train_ProtA = open("dataAfterPreProcessing/PPI_Dataset/trainingSequencesDataSetProteinA_fromSet.fasta", "w")
    Train_ProtB = open("dataAfterPreProcessing/PPI_Dataset/trainingSequencesDataSetProteinB_fromSet.fasta", "w")
    for pair in trainInteractingPair_Seqs:
        Train_ProtA.write(">" + pair[0] + " \n")
        Train_ProtA.write(pair[1] + " \n")
        Train_ProtB.write(">" + pair[2] + " \n")
        Train_ProtB.write(pair[3] + " \n")

    # write the test dataset (set) as a FASTA file:
    Test_ProtA = open("dataAfterPreProcessing/PPI_Dataset/testSequencesDataSetProteinA_fromSet.fasta", "w")
    Test_ProtB = open("dataAfterPreProcessing/PPI_Dataset/testSequencesDataSetProteinB_fromSet.fasta", "w")
    for pair in testInteractingPair_Seqs:
        Test_ProtA.write(">" + pair[0] + " \n")
        Test_ProtA.write(pair[1] + " \n")
        Test_ProtB.write(">" + pair[2] + " \n")
        Test_ProtB.write(pair[3] + " \n")
    print("*********************************************************************")

    print("** Blast the Test dataset ProteinA against the train dataset ProteinA using blastp\n")
    f.write("** Blast the Test dataset ProteinA against the train dataset ProteinA using blastp\n")
    # Note:
    # Blasting using a system call
    os.system("./blast-2.15.0+/bin/makeblastdb -in trainingSequencesDataSetProteinA.fasta -title TrainingDB_ProtA -dbtype prot ")
    # os.system("./blast-2.15.0+/bin/makeblastdb -in trainingSequencesDataSetProteinA.fasta -parse_seqids -title TrainingDB_ProtA -dbtype prot ")
    os.system("./blast-2.15.0+/bin/blastp -query testSequencesDataSetProteinA.fasta -out testBlastedTrainProtA.csv -db trainingSequencesDataSetProteinA.fasta -outfmt 6 -num_threads 4 ")

    print("** Blast the Test dataset ProteinB against the train dataset ProteinB using blastp\n")
    f.write("** Blast the Test dataset ProteinB against the train dataset ProteinB using blastp\n")
    os.system("./blast-2.15.0+/bin/makeblastdb -in trainingSequencesDataSetProteinB.fasta -title TrainingDB_ProtB -dbtype prot ")
    os.system("./blast-2.15.0+/bin/blastp -query testSequencesDataSetProteinB.fasta -out testBlastedTrainProtB.csv -db trainingSequencesDataSetProteinB.fasta -outfmt 6 -num_threads 4 ")

    print("**********************************************************************")
    print("**** Parsing the blast results\n")
    f.write("**** Parsing the blast results\n")
    # Read the blast results CVS files:
    testBlastResult_ProtA = pd.read_csv("testBlastedTrainProtA.csv", sep="\t", header=None)
    testBlastResult_ProtB = pd.read_csv("testBlastedTrainProtB.csv", sep="\t", header=None)
    # print("hello")

    # constract a dictionary that contians the homology percentage perprotein of TestDataSet_ProtA
    maxHomologyPerTestProteinA=dict()
    for rowIndx in range(testBlastResult_ProtA.shape[0]):
        if testBlastResult_ProtA.iloc[rowIndx,0] in maxHomologyPerTestProteinA.keys():
            maxHomologyPerTestProteinA[testBlastResult_ProtA.iloc[rowIndx,0]]=max(
                    maxHomologyPerTestProteinA[testBlastResult_ProtA.iloc[rowIndx,0]],
                    testBlastResult_ProtA.iloc[rowIndx,2])
        else:
            maxHomologyPerTestProteinA[testBlastResult_ProtA.iloc[rowIndx,0]]=testBlastResult_ProtA.iloc[rowIndx,2]
    print("constract a dictionary that contians the homology percentage perprotein of TestDataSet_ProtA\n")
    f.write("constract a dictionary that contians the homology percentage perprotein of TestDataSet_ProtA\n")

    # Extract proteins of TestDataSet_ProtA that have homology of 40% and less:
    cleanedByHomologyTestDataSet_ProtA=dict()
    for proteinId, homologyRatio in maxHomologyPerTestProteinA.items():
        if homologyRatio<40:
            cleanedByHomologyTestDataSet_ProtA[proteinId]=homologyRatio
    testDataSetUniqueIds_ProtA=cleanedByHomologyTestDataSet_ProtA.keys()

    print("Extract proteins of TestDataSet_ProtA that have homology of 40% and less:\n")
    f.write("Extract proteins of TestDataSet_ProtA that have homology of 40% and less:\n")

    # constract a dictionary that contians the homology percentage perprotein of TestDataSet_ProtB
    maxHomologyPerTestProteinB=dict()
    for rowIndx in range(testBlastResult_ProtB.shape[0]):
        if testBlastResult_ProtB.iloc[rowIndx,0] in maxHomologyPerTestProteinB.keys():
            maxHomologyPerTestProteinB[testBlastResult_ProtB.iloc[rowIndx,0]]=max(
                    maxHomologyPerTestProteinB[testBlastResult_ProtB.iloc[rowIndx,0]],
                    testBlastResult_ProtB.iloc[rowIndx,2])
        else:
            maxHomologyPerTestProteinB[testBlastResult_ProtB.iloc[rowIndx,0]]=testBlastResult_ProtB.iloc[rowIndx,2]

    print("constract a dictionary that contians the homology percentage perprotein of TestDataSet_ProtB\n")
    f.write("constract a dictionary that contians the homology percentage perprotein of TestDataSet_ProtB\n")

    # Extract proteins of TestDataSet_ProtA that have homology of 40% and less:
    cleanedByHomologyTestDataSet_ProtB=dict()
    for proteinId, homologyRatio in maxHomologyPerTestProteinB.items():
        if homologyRatio<40:
            cleanedByHomologyTestDataSet_ProtB[proteinId]=homologyRatio
    testDataSetUniqueIds_ProtB=cleanedByHomologyTestDataSet_ProtB.keys()

    print("Extract proteins of TestDataSet_ProtA that have homology of 40% and less:\n")
    f.write("Extract proteins of TestDataSet_ProtA that have homology of 40% and less:\n")

    testInteractingPair_Seqs_afterRemoveHomology = set()
    for pair in testInteractingPair_Seqs:
        if (pair[0] in testDataSetUniqueIds_ProtA and pair[2] in testDataSetUniqueIds_ProtB):
            testInteractingPair_Seqs_afterRemoveHomology.add(pair)

    ## Writing the testInteractingPair_Seqs_afterRemoveHomology to the disk:
    with open("dataAfterPreProcessing/PPI_Dataset/testInteractingPair_Seqs_afterRemoveHomology.pickle", "wb") as output_:
        pickle.dump(testInteractingPair_Seqs_afterRemoveHomology, output_)
    print("Writing the testInteractingPair_Seqs_afterRemoveHomology to the disk\n")
    f.write("Writing the testInteractingPair_Seqs_afterRemoveHomology to the disk\n")
    print("number of testInteractingPair_Seqs_afterRemoveHomology is : " + str(len(testInteractingPair_Seqs_afterRemoveHomology)) + "\n")
    f.write("number of testInteractingPair_Seqs_afterRemoveHomology is : " + str(len(testInteractingPair_Seqs_afterRemoveHomology)) + "\n")

    PPI_Final_UniqueProteinIDs = set() # these protein will be removed from the Proteins dataset which will used for pre-training the model
    for pair in trainInteractingPair_Seqs:
        PPI_Final_UniqueProteinIDs.add(pair[0])
        PPI_Final_UniqueProteinIDs.add(pair[2])

    for pair in testInteractingPair_Seqs_afterRemoveHomology:
        PPI_Final_UniqueProteinIDs.add(pair[0])
        PPI_Final_UniqueProteinIDs.add(pair[2])

    ## Writing the PPI_Final_UniqueProteinIDs to the disk:
    with open("dataAfterPreProcessing/PPI_Dataset/PPI_Final_UniqueProteinIDs.pickle", "wb") as output_:
        pickle.dump(PPI_Final_UniqueProteinIDs, output_)
    print("## Writing the PPI_Final_UniqueProteinIDs to the disk:\n")
    f.write("## Writing the PPI_Final_UniqueProteinIDs to the disk:\n")

    print("number of PPI_Final_UniqueProteinIDs is : " + str(len(PPI_Final_UniqueProteinIDs)) + "\n")
    f.write("number of PPI_Final_UniqueProteinIDs is : " + str(len(PPI_Final_UniqueProteinIDs)) + "\n")

    # trainInteractingPair_Seqs
    # store the first_ProteinPos and the secondProteinPos of the train and test datasets seperatly
    firstProteinTrainPos, secondProteinTrainPos = [], []
    for pair in trainInteractingPair_Seqs:
        firstProteinTrainPos.append((pair[0], pair[1]))
        secondProteinTrainPos.append((pair[2], pair[3]))

    assert len(firstProteinTrainPos) == len(secondProteinTrainPos)
    labelsPositiveTrain = np.ones((len(firstProteinTrainPos))).reshape(-1, 1)

    f.write("len of firstProteinTrainPos : " + str(len(firstProteinTrainPos)) + "\n")
    f.write("len of secondProteinTrainPos : " + str(len(secondProteinTrainPos)) + "\n")
    f.write("len of labelsPositiveTrain : " + str(len(labelsPositiveTrain)) + "\n")

    f.write("store the test dataset locally\n")
    with open("dataAfterPreProcessing/PPI_Dataset/firstProteinTrainPos.pickle", "wb") as output_:
        pickle.dump(firstProteinTrainPos, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/secondProteinTrainPos.pickle", "wb") as output_:
        pickle.dump(secondProteinTrainPos, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/labelsPositiveTrain.pickle", "wb") as output_:
        pickle.dump(labelsPositiveTrain, output_)

    # construct negative examples of the training dataset:
    firstProteinTrainNeg, secondProteinTrainNeg = [], []
    for _ in range(len(firstProteinTrainPos)):
        firstProteinTrainNeg.append(sample(firstProteinTrainPos, 1)[0])
        secondProteinTrainNeg.append(sample(secondProteinTrainPos, 1)[0])
    assert len(firstProteinTrainNeg) == len(secondProteinTrainNeg)
    labelsNegativeTrain = np.zeros(len(firstProteinTrainNeg)).reshape(-1, 1)

    f.write("len of firstProteinTrainNeg : " + str(len(firstProteinTrainNeg)) + "\n")
    f.write("len of secondProteinTrainNeg : " + str(len(secondProteinTrainNeg)) + "\n")
    f.write("len of labelsNegativeTrain : " + str(len(labelsNegativeTrain)) + "\n")

    f.write("store the test dataset locally\n")
    with open("dataAfterPreProcessing/PPI_Dataset/firstProteinTrainNeg.pickle", "wb") as output_:
        pickle.dump(firstProteinTrainNeg, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/secondProteinTrainNeg.pickle", "wb") as output_:
        pickle.dump(secondProteinTrainNeg, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/labelsNegativeTrain.pickle", "wb") as output_:
        pickle.dump(labelsNegativeTrain, output_)

    # assemble the training DataSet:
    firstProteinsListTrain = firstProteinTrainPos + firstProteinTrainNeg
    secondProteinsListTrain = secondProteinTrainPos + secondProteinTrainNeg
    thirdProteinsTensorTrain=np.concatenate((labelsPositiveTrain,labelsNegativeTrain),
                                   axis=0)
    assert len(firstProteinsListTrain)==thirdProteinsTensorTrain.shape[0]
    assert len(secondProteinsListTrain)==thirdProteinsTensorTrain.shape[0]

    f.write("len of firstProteinsListTrain : " + str(len(firstProteinsListTrain)) + "\n")
    f.write("len of secondProteinsListTrain : " + str(len(secondProteinsListTrain)) + "\n")
    f.write("len of thirdProteinsTensorTrain : " + str(len(thirdProteinsTensorTrain)) + "\n")

    f.write("store the test dataset locally\n")
    with open("dataAfterPreProcessing/PPI_Dataset/firstProteinsListTrain.pickle", "wb") as output_:
        pickle.dump(firstProteinsListTrain, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/secondProteinsListTrain.pickle", "wb") as output_:
        pickle.dump(secondProteinsListTrain, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/thirdProteinsTensorTrain.pickle", "wb") as output_:
        pickle.dump(thirdProteinsTensorTrain, output_)

    # testInteractingPair_Seqs_afterRemoveHomology
    firstProteinTestPos, secondProteinTestPos = [], []
    for pair in testInteractingPair_Seqs_afterRemoveHomology:
        firstProteinTestPos.append((pair[0], pair[1]))
        secondProteinTestPos.append((pair[2], pair[3]))

    assert len(firstProteinTestPos) == len(secondProteinTestPos)
    labelsPositiveTest = np.ones((len(firstProteinTestPos))).reshape(-1, 1)


    f.write("len of firstProteinTestPos : " + str(len(firstProteinTestPos)) + "\n")
    f.write("len of secondProteinTestPos : " + str(len(secondProteinTestPos)) + "\n")
    f.write("len of labelsPositiveTest : " + str(len(labelsPositiveTest)) + "\n")

    f.write("store the test dataset locally\n")
    with open("dataAfterPreProcessing/PPI_Dataset/firstProteinTestPos.pickle", "wb") as output_:
        pickle.dump(firstProteinTestPos, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/secondProteinTestPos.pickle", "wb") as output_:
        pickle.dump(secondProteinTestPos, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/labelsPositiveTest.pickle", "wb") as output_:
        pickle.dump(labelsPositiveTest, output_)

    # construct negative examples of the test dataset:
    firstProteinTestNeg, secondProteinTestNeg = [], []
    for _ in range(len(firstProteinTestPos)):
        firstProteinTestNeg.append(sample(firstProteinTestPos, 1)[0])
        secondProteinTestNeg.append(sample(secondProteinTestPos, 1)[0])
    assert len(firstProteinTestNeg) == len(secondProteinTestNeg)
    labelsNegativeTest = np.zeros(len(firstProteinTestNeg)).reshape(-1, 1)

    f.write("len of firstProteinTestNeg : " + str(len(firstProteinTestNeg)) + "\n")
    f.write("len of secondProteinTestNeg : " + str(len(secondProteinTestNeg)) + "\n")
    f.write("len of labelsNegativeTest : " + str(len(labelsNegativeTest)) + "\n")

    f.write("store the test dataset locally\n")
    with open("dataAfterPreProcessing/PPI_Dataset/firstProteinTestNeg.pickle", "wb") as output_:
        pickle.dump(firstProteinTestNeg, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/secondProteinTestNeg.pickle", "wb") as output_:
        pickle.dump(secondProteinTestNeg, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/labelsNegativeTest.pickle", "wb") as output_:
        pickle.dump(labelsNegativeTest, output_)

    # assemble the test DataSet:
    firstProteinsListTest = firstProteinTestPos + firstProteinTestNeg
    secondProteinsListTest = secondProteinTestPos + secondProteinTestNeg
    thirdProteinsTensorTest=np.concatenate((labelsPositiveTest,labelsNegativeTest),
                                   axis=0)

    assert len(firstProteinsListTest)==thirdProteinsTensorTest.shape[0]
    assert len(secondProteinsListTest)==thirdProteinsTensorTest.shape[0]

    f.write("len of firstProteinsListTest : " + str(len(firstProteinsListTest)) + "\n")
    f.write("len of secondProteinsListTest : " + str(len(secondProteinsListTest)) + "\n")
    f.write("len of thirdProteinsTensorTest : " + str(len(thirdProteinsTensorTest)) + "\n")

    f.write("store the test dataset locally\n")
    with open("dataAfterPreProcessing/PPI_Dataset/firstProteinsListTest.pickle", "wb") as output_:
        pickle.dump(firstProteinsListTest, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/secondProteinsListTest.pickle", "wb") as output_:
        pickle.dump(secondProteinsListTest, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/thirdProteinsTensorTest.pickle", "wb") as output_:
        pickle.dump(thirdProteinsTensorTest, output_)
