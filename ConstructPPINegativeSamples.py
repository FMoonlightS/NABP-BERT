"""
@author: Fatma S. Ahmed
@brief: Construct PPI Negative Samples.
"""

import pickle
import random
from random import sample
# from utils import shuffleArray, splitSequenceintoKmers
import numpy as np

random.seed(123)
with open('ConstructNegativeSamples.txt','w') as f:

    # load trainInteractingPair_Seqs
    with open("dataAfterPreProcessing/PPI_Dataset/trainInteractingPair_Seqs_set.pickle", "rb") as output_:
        trainInteractingPair_Seqs = pickle.load(output_)
    print("number of trainInteractingPair_Seqs is : " + str(len(trainInteractingPair_Seqs))+ "\n")
    f.write("number of trainInteractingPair_Seqs is : " + str(len(trainInteractingPair_Seqs))+ "\n")

    # store the first_ProteinPos and the secondProteinPos of the train and test datasets separately
    firstProteinTrainPos, secondProteinTrainPos = [], []
    for pair in trainInteractingPair_Seqs:
        firstProteinTrainPos.append((pair[0], pair[1]))
        secondProteinTrainPos.append((pair[2], pair[3]))

    assert len(firstProteinTrainPos) == len(secondProteinTrainPos)
    labelsPositiveTrain = np.ones((len(firstProteinTrainPos))).reshape(-1, 1)

    f.write("len of firstProteinTrainPos : " + str(len(firstProteinTrainPos)) + "\n")
    f.write("len of secondProteinTrainPos : " + str(len(secondProteinTrainPos)) + "\n")
    f.write("len of labelsPositiveTrain : " + str(len(labelsPositiveTrain)) + "\n")

    f.write("store the train dataset locally\n")
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

    f.write("store the train dataset locally\n")
    with open("dataAfterPreProcessing/PPI_Dataset/firstProteinTrainNeg.pickle", "wb") as output_:
        pickle.dump(firstProteinTrainNeg, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/secondProteinTrainNeg.pickle", "wb") as output_:
        pickle.dump(secondProteinTrainNeg, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/labelsNegativeTrain.pickle", "wb") as output_:
        pickle.dump(labelsNegativeTrain, output_)

    # assemble the training DataSet:
    firstProteinsListTrain = firstProteinTrainPos + firstProteinTrainNeg
    secondProteinsListTrain = secondProteinTrainPos + secondProteinTrainNeg
    thirdProteinsTensorTrain = np.concatenate((labelsPositiveTrain, labelsNegativeTrain),
                                              axis=0)
    assert len(firstProteinsListTrain) == thirdProteinsTensorTrain.shape[0]
    assert len(secondProteinsListTrain) == thirdProteinsTensorTrain.shape[0]

    f.write("len of firstProteinsListTrain : " + str(len(firstProteinsListTrain)) + "\n")
    f.write("len of secondProteinsListTrain : " + str(len(secondProteinsListTrain)) + "\n")
    f.write("len of thirdProteinsTensorTrain : " + str(len(thirdProteinsTensorTrain)) + "\n")

    f.write("store the train dataset locally\n")
    with open("dataAfterPreProcessing/PPI_Dataset/firstProteinsListTrain.pickle", "wb") as output_:
        pickle.dump(firstProteinsListTrain, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/secondProteinsListTrain.pickle", "wb") as output_:
        pickle.dump(secondProteinsListTrain, output_)

    with open("dataAfterPreProcessing/PPI_Dataset/thirdProteinsTensorTrain.pickle", "wb") as output_:
        pickle.dump(thirdProteinsTensorTrain, output_)

    # load testInteractingPair_Seqs_afterRemoveHomology
    with open("dataAfterPreProcessing/PPI_Dataset/testInteractingPair_Seqs_afterRemoveHomology.pickle", "rb") as output_:
        testInteractingPair_Seqs_afterRemoveHomology = pickle.load(output_)
    print("number of testInteractingPair_Seqs_afterRemoveHomology is : " + str(len(testInteractingPair_Seqs_afterRemoveHomology))+ "\n")
    f.write("number of testInteractingPair_Seqs_afterRemoveHomology is : " + str(len(testInteractingPair_Seqs_afterRemoveHomology))+ "\n")

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
    thirdProteinsTensorTest = np.concatenate((labelsPositiveTest, labelsNegativeTest),
                                             axis=0)
    assert len(firstProteinsListTest) == thirdProteinsTensorTest.shape[0]
    assert len(secondProteinsListTest) == thirdProteinsTensorTest.shape[0]

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
