"""
@author: Fatma S. Ahmed
@brief: Prepear PreTrain Dataset.
"""

import pickle
import random
from random import sample

# import gc

def sequence2tokens(seq):
    tokens = []
    for i in range(len(seq) - 2):
        tokens.append(seq[i:i + 3])
    return tokens


with open('Prepear_PreTrain_Dataset_3_3.txt','w') as f:
    # get all Protein Sequences from Swissport (PretrainData_Swissport) that have a lenght less than 511 which will produce 509 3-mer tokens
    with open("dataAfterPreProcessing/SwissDataBase/SwissDataBase.pickle","rb") as input_:
        database_Swiss=pickle.load(input_)

    print("size of Swissport database : " + str(len(database_Swiss)) + "\n")
    f.write("size of Swissport database : " + str(len(database_Swiss)) + "\n")

    PretrainData_Swissport = set()
    SwissKeys = database_Swiss.keys()
    for key in SwissKeys:
        # if len(database_Swiss[key]) <= 511:
        if (len(database_Swiss[key]) >= 100 and len(database_Swiss[key]) <= 511):
            PretrainData_Swissport.add((key, database_Swiss[key]))
    print("size of PretrainData_Swissport database : " + str(len(PretrainData_Swissport)) + "\n")
    f.write("size of PretrainData_Swissport database : " + str(len(PretrainData_Swissport)) + "\n")

    print("store PretrainData_Swissport locally\n")
    f.write("store PretrainData_Swissport locally\n")
    with open("dataAfterPreProcessing/PreTrainDataset/PretrainData_Swissport.pickle","wb") as output_:
        pickle.dump(PretrainData_Swissport,output_)

    # load PPI_Final_UniqueProteinIDs
    with open("dataAfterPreProcessing/PPI_Dataset/PPI_Final_UniqueProteinIDs.pickle","rb") as input_:
        PPI_Final_UniqueProteinIDs=pickle.load(input_)
    print("size of PPI_Final_UniqueProteinIDs  : " + str(len(PPI_Final_UniqueProteinIDs)) + "\n")
    f.write("size of PPI_Final_UniqueProteinIDs  : " + str(len(PPI_Final_UniqueProteinIDs)) + "\n")

    PretrainData_Swissport_AfterRemoveExist = set()
    j = 0
    for item in PretrainData_Swissport:
        if item[0] in PPI_Final_UniqueProteinIDs:
            j += 1
            continue
        else:
            PretrainData_Swissport_AfterRemoveExist.add(item)

    print("number of Protein IDs exists in PretrainData_Swissport : " + str(j) + "\n")
    f.write("number of Protein IDs exists in PretrainData_Swissport : " + str(j) + "\n")

    print("size of PretrainData_Swissport_AfterRemoveExist  : " + str(len(PretrainData_Swissport_AfterRemoveExist)) + "\n")
    f.write("size of PretrainData_Swissport_AfterRemoveExist  : " + str(len(PretrainData_Swissport_AfterRemoveExist)) + "\n")

    with open("dataAfterPreProcessing/PreTrainDataset/PretrainData_Swissport_AfterRemoveExist.pickle","wb") as output_:
        pickle.dump(PretrainData_Swissport_AfterRemoveExist,output_)
###################################################################################

    # get all Protein Sequences from Tremble (PretrainData_Tremble) that have a lenght less than 511 which will produce 509 3-mer tokens
    PretrainData_Tremble = set()
    i = 51
    # for i in range(1, 52):
        # with open("dataAfterPreProcessing/TrembleDataBase/TrembleDataBase_" + str(i) + ".pickle", "rb") as input_:
    with open("dataAfterPreProcessing/TrembleDataBase/TrembleDataBase_" + str(i) + ".pickle", "rb") as input_:
        TrembleDataBase = pickle.load(input_)

    print("size of TrembleDataBase database : " + str(len(TrembleDataBase)) + "\n")
    f.write("size of TrembleDataBase database : " + str(len(TrembleDataBase)) + "\n")

    TrembleKeys = TrembleDataBase.keys()
    for key in TrembleKeys:
        # if len(TrembleDataBase[key]) <= 511:
        if (len(TrembleDataBase[key]) >= 100 and len(TrembleDataBase[key]) <= 511):
            PretrainData_Tremble.add((key, TrembleDataBase[key]))

    print("size of PretrainData_Tremble database : " + str(len(PretrainData_Tremble)) + "\n")
    f.write("size of PretrainData_Tremble database : " + str(len(PretrainData_Tremble)) + "\n")

    print("store PretrainData_Tremble locally\n")
    f.write("store PretrainData_Tremble locally\n")
    with open("dataAfterPreProcessing/PreTrainDataset/PretrainData_Tremble.pickle","wb") as output_:
        pickle.dump(PretrainData_Tremble,output_)

    PretrainData_Tremble__AfterRemoveExist = set()
    k = 0
    for item in PretrainData_Tremble:
        if item[0] in PPI_Final_UniqueProteinIDs:
            k += 1
            continue
        else:
            PretrainData_Tremble__AfterRemoveExist.add(item)

    print("number of Protein IDs exists in PretrainData_Tremble : " + str(k) + "\n")
    f.write("number of Protein IDs exists in PretrainData_Tremble : " + str(k) + "\n")

    print("size of PretrainData_Tremble__AfterRemoveExist  : " + str(len(PretrainData_Tremble__AfterRemoveExist)) + "\n")
    f.write("size of PretrainData_Tremble__AfterRemoveExist  : " + str(len(PretrainData_Tremble__AfterRemoveExist)) + "\n")

    with open("dataAfterPreProcessing/PreTrainDataset/PretrainData_Tremble__AfterRemoveExist.pickle","wb") as output_:
        pickle.dump(PretrainData_Tremble__AfterRemoveExist,output_)

################################################################################
    PretrainData_Final = set()

    # combine PretrainData_Swissport and PretrainData_Tremble ===> PretrainData
    for item in PretrainData_Swissport_AfterRemoveExist:
        PretrainData_Final.add(item)
    # for item in PretrainData_Tremble__AfterRemoveExist:
    #     PretrainData_Final.add(item)

    for item in PretrainData_Tremble__AfterRemoveExist:
        PretrainData_Final.add(item)
        if (len(PretrainData_Final) == len(PretrainData_Swissport_AfterRemoveExist)*2):
            break

    print("size of PretrainData_Final database : " + str(len(PretrainData_Final)) + "\n")
    f.write("size of PretrainData_Final database : " + str(len(PretrainData_Final)) + "\n")

    print("store PretrainData_Final locally\n")
    f.write("store PretrainData_Final locally\n")
    with open("dataAfterPreProcessing/PreTrainDataset/PretrainData_Final.pickle", "wb") as output_:
        pickle.dump(PretrainData_Final, output_)

    # write the each sequence in PretrainData_Final as 3-mer tokens in .txt file
    with open("dataAfterPreProcessing/PreTrainDataset/PretrainData_Final.txt", "w") as output_:
        for item in PretrainData_Final:
            tokens = sequence2tokens(item[1])
            for token in tokens:
                output_.write(token + " ")
            output_.write("\n\n")
f.close()
