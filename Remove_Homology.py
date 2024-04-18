"""
@author: Fatma S. Ahmed
@brief: Remove Homology.
"""

import pickle
import pandas as pd
with open('Remove_Homology.txt','w') as f:
    print("**** Parsing the blast results\n")
    f.write("**** Parsing the blast results\n")
    # Read the blast results CVS files:
    testBlastResult_ProtA = pd.read_csv("testBlastedTrainProtA.csv", sep="\t", header=None)
    testBlastResult_ProtB = pd.read_csv("testBlastedTrainProtB.csv", sep="\t", header=None)
    # print("hello")

    # constract a dictionary that contians the homology percentage perprotein of TestDataSet_ProtA
    maxHomologyPerTestProteinA = dict()
    for rowIndx in range(testBlastResult_ProtA.shape[0]):
        if testBlastResult_ProtA.iloc[rowIndx, 0] in maxHomologyPerTestProteinA.keys():
            maxHomologyPerTestProteinA[testBlastResult_ProtA.iloc[rowIndx, 0]] = max(
                maxHomologyPerTestProteinA[testBlastResult_ProtA.iloc[rowIndx, 0]],
                testBlastResult_ProtA.iloc[rowIndx, 2])
        else:
            maxHomologyPerTestProteinA[testBlastResult_ProtA.iloc[rowIndx, 0]] = testBlastResult_ProtA.iloc[rowIndx, 2]
    print("constract a dictionary that contians the homology percentage perprotein of TestDataSet_ProtA\n")
    f.write("constract a dictionary that contians the homology percentage perprotein of TestDataSet_ProtA\n")

    # Extract proteins of TestDataSet_ProtA that have homology of 40% and less:
    cleanedByHomologyTestDataSet_ProtA = dict()
    for proteinId, homologyRatio in maxHomologyPerTestProteinA.items():
        if homologyRatio < 40:
            cleanedByHomologyTestDataSet_ProtA[proteinId] = homologyRatio
    testDataSetUniqueIds_ProtA = cleanedByHomologyTestDataSet_ProtA.keys()

    print("Extract proteins of TestDataSet_ProtA that have homology of 40% and less:\n")
    f.write("Extract proteins of TestDataSet_ProtA that have homology of 40% and less:\n")

    # constract a dictionary that contians the homology percentage perprotein of TestDataSet_ProtB
    maxHomologyPerTestProteinB = dict()
    for rowIndx in range(testBlastResult_ProtB.shape[0]):
        if testBlastResult_ProtB.iloc[rowIndx, 0] in maxHomologyPerTestProteinB.keys():
            maxHomologyPerTestProteinB[testBlastResult_ProtB.iloc[rowIndx, 0]] = max(
                maxHomologyPerTestProteinB[testBlastResult_ProtB.iloc[rowIndx, 0]],
                testBlastResult_ProtB.iloc[rowIndx, 2])
        else:
            maxHomologyPerTestProteinB[testBlastResult_ProtB.iloc[rowIndx, 0]] = testBlastResult_ProtB.iloc[rowIndx, 2]

    print("constract a dictionary that contians the homology percentage perprotein of TestDataSet_ProtB\n")
    f.write("constract a dictionary that contians the homology percentage perprotein of TestDataSet_ProtB\n")

    # Extract proteins of TestDataSet_ProtA that have homology of 40% and less:
    cleanedByHomologyTestDataSet_ProtB = dict()
    for proteinId, homologyRatio in maxHomologyPerTestProteinB.items():
        if homologyRatio < 40:
            cleanedByHomologyTestDataSet_ProtB[proteinId] = homologyRatio
    testDataSetUniqueIds_ProtB = cleanedByHomologyTestDataSet_ProtB.keys()

    print("Extract proteins of TestDataSet_ProtA that have homology of 40% and less:\n")
    f.write("Extract proteins of TestDataSet_ProtA that have homology of 40% and less:\n")

    with open("dataAfterPreProcessing/PPI_Dataset/trainInteractingPair_Seqs_set.pickle", "rb") as output_:
        trainInteractingPair_Seqs = pickle.load(output_)
    print("number of trainInteractingPair_Seqs is : " + str(len(trainInteractingPair_Seqs))+ "\n")
    f.write("number of trainInteractingPair_Seqs is : " + str(len(trainInteractingPair_Seqs))+ "\n")

    with open("dataAfterPreProcessing/PPI_Dataset/testInteractingPair_Seqs_set.pickle", "rb") as output_:
        testInteractingPair_Seqs = pickle.load(output_)
    print("number of testInteractingPair_Seqs is : " + str(len(testInteractingPair_Seqs))+ "\n")
    f.write("number of testInteractingPair_Seqs is : " + str(len(testInteractingPair_Seqs))+ "\n")

    testInteractingPair_Seqs_afterRemoveHomology = set()
    for pair in testInteractingPair_Seqs:
        if (pair[0] in testDataSetUniqueIds_ProtA or pair[2] in testDataSetUniqueIds_ProtB):
            testInteractingPair_Seqs_afterRemoveHomology.add(pair)

    ## Writing the testInteractingPair_Seqs_afterRemoveHomology to the disk:
    with open("dataAfterPreProcessing/PPI_Dataset/testInteractingPair_Seqs_afterRemoveHomology.pickle",
              "wb") as output_:
        pickle.dump(testInteractingPair_Seqs_afterRemoveHomology, output_)
    print("Writing the testInteractingPair_Seqs_afterRemoveHomology to the disk\n")
    f.write("Writing the testInteractingPair_Seqs_afterRemoveHomology to the disk\n")
    print("number of testInteractingPair_Seqs_afterRemoveHomology is : " + str(
        len(testInteractingPair_Seqs_afterRemoveHomology)) + "\n")
    f.write("number of testInteractingPair_Seqs_afterRemoveHomology is : " + str(
        len(testInteractingPair_Seqs_afterRemoveHomology)) + "\n")

    PPI_Final_UniqueProteinIDs = set()  # these protein will be removed from the Proteins dataset which will used for pre-training the model
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
