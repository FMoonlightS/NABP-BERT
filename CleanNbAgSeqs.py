"""
@author: Fatma S. Ahmed
@brief: Cleaning Nanobody-Antigen Sequences.
"""

# Nb (min , max) len is (104 , 175)
# Ag (min , max) len is (158 , 1816)
# Nb max seq len is 175 and this will produce 173 token at 3-mer,
# so the Ag max seq len should be 388 which will produc 386 tokens at 3-mers,
# as a result when conbine Nb-Ag pairs for BERT model their len will be 509,
# Bert max len is 512 = 509 + 3

import pickle
Cleaned_Nb_Ag_pairs = []  # this list will contain the pairs which their antigen seq len <=338
with open('CleanNbAgSeqs.txt','w') as f:

    with open("dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/Nb_Ag_full_dataset.pickle", "rb") as output_:
        Nb_Ag_full_dataset = pickle.load(output_)
    # Nb_Ag_full_dataset is a list of 3506 Nb-Ag pairs,
    # each pair represent by a tuple (Nb_seq, Ag_seq, Ag_uniprot_id, label)
    #convert the lables from Yes and No to 1 and 0
    Nb_Ag_full_dataset_2 = []
    pos_pairs_beforeCleaning = 0
    neg_pairs_beforeCleaning = 0
    for pair in Nb_Ag_full_dataset:
        if pair[3] == "Yes":
            Nb_Ag_full_dataset_2.append((pair[0],pair[1],pair[2],1))
            pos_pairs_beforeCleaning += 1
        else:
            Nb_Ag_full_dataset_2.append((pair[0],pair[1],pair[2],0))
            neg_pairs_beforeCleaning += 1

    with open('dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/Nb_Ag_full_dataset_2.pickle','wb') as binary_writer:
        pickle.dump(Nb_Ag_full_dataset_2,binary_writer)

    pos_pairs_afterCleaning = 0
    Cleaned_Nb_Ag_pairs_unique_UniProtIDs = set()
    for pair in Nb_Ag_full_dataset_2:
        if len(pair[1]) <= 338:
            Cleaned_Nb_Ag_pairs.append(pair)
            Cleaned_Nb_Ag_pairs_unique_UniProtIDs.add(pair[2])
            pos_pairs_afterCleaning += pair[3]

    print("Total pairs before Cleaning Nb_Ag_pairs : " + str(len(Nb_Ag_full_dataset_2)) + "\n")
    print("pos pairs of Uncleaned_Nb_Ag_pairs : " + str(pos_pairs_beforeCleaning) + "\n")
    print("neg pairs of Uncleaned_Nb_Ag_pairs : " + str(neg_pairs_beforeCleaning) + "\n")

    print("Total pairs after Cleaning Nb_Ag_pairs : " + str(len(Cleaned_Nb_Ag_pairs)) + "\n")
    print("pos pairs of Cleaned_Nb_Ag_pairs : " + str(pos_pairs_afterCleaning) + "\n")
    print("neg pairs of Cleaned_Nb_Ag_pairs : " + str(len(Cleaned_Nb_Ag_pairs)-pos_pairs_afterCleaning) + "\n")
    print("Number of Cleaned_Nb_Ag_pairs_unique_UniProtIDs : " + str(len(Cleaned_Nb_Ag_pairs_unique_UniProtIDs)) + "\n")

    f.write("Total pairs before Cleaning Nb_Ag_pairs : " + str(len(Nb_Ag_full_dataset_2)) + "\n")
    f.write("pos pairs of Uncleaned_Nb_Ag_pairs : " + str(pos_pairs_beforeCleaning) + "\n")
    f.write("neg pairs of Uncleaned_Nb_Ag_pairs : " + str(neg_pairs_beforeCleaning) + "\n")

    f.write("Total pairs after Cleaning Nb_Ag_pairs : " + str(len(Cleaned_Nb_Ag_pairs)) + "\n")
    f.write("pos pairs of Cleaned_Nb_Ag_pairs : " + str(pos_pairs_afterCleaning) + "\n")
    f.write("neg pairs of Cleaned_Nb_Ag_pairs : " + str(len(Cleaned_Nb_Ag_pairs)-pos_pairs_afterCleaning) + "\n")
    f.write("Number of Cleaned_Nb_Ag_pairs_unique_UniProtIDs : " + str(len(Cleaned_Nb_Ag_pairs_unique_UniProtIDs)) + "\n")

    f.write("Saving the cleaned Nb-Ag pairs database locally \n")
    print("Saving the cleaned Nb-Ag pairs database locally \n")

    with open('dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/Cleaned_Nb_Ag_pairs.pickle','wb') as binary_writer:
        pickle.dump(Cleaned_Nb_Ag_pairs,binary_writer)


    f.write("Saving the Cleaned_Nb_Ag_pairs_unique_UniProtIDs locally \n")
    print("Saving the Cleaned_Nb_Ag_pairs_unique_UniProtIDs locally \n")

    with open('dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/Cleaned_Nb_Ag_pairs_unique_UniProtIDs.pickle','wb') as binary_writer:
        pickle.dump(Cleaned_Nb_Ag_pairs_unique_UniProtIDs,binary_writer)