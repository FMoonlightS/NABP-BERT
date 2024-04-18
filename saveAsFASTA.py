"""
@author: Fatma S. Ahmed
@brief: save As FASTA file.
"""

import pickle
# with open('./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/Cleaned_Nb_Ag_pairs.pickle','rb') as binary_reader:
#     Cleaned_Nb_Ag_pairs = pickle.load(binary_reader)
# nanobody = open("./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/nanobody.fasta","w")
# antigen = open("./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/antigen.fasta","w")
# lables = open("./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/labels.txt","w")

# with open('./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/train2Cleaned_Nb_Ag_pairs.pickle','rb') as binary_reader:
#     Cleaned_Nb_Ag_pairs = pickle.load(binary_reader)
# nanobody = open("./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/nanobody_train.fasta","w")
# antigen = open("./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/antigen_train.fasta","w")
# lables = open("./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/labels_train.txt","w")

# with open('./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/valCleaned_Nb_Ag_pairs.pickle','rb') as binary_reader:
#     Cleaned_Nb_Ag_pairs = pickle.load(binary_reader)
# nanobody = open("./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/nanobody_val.fasta","w")
# antigen = open("./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/antigen_val.fasta","w")
# lables = open("./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/labels_val.txt","w")

with open('./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/testCleaned_Nb_Ag_pairs.pickle','rb') as binary_reader:
    Cleaned_Nb_Ag_pairs = pickle.load(binary_reader)
nanobody = open("./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/nanobody_test.fasta","w")
antigen = open("./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/antigen_test.fasta","w")
lables = open("./dataAfterPreProcessing/Nb_Ag_Pairs_Dataset/labels_test.txt","w")

for pair in Cleaned_Nb_Ag_pairs:
    nanobody.write(">\n")
    nanobody.write(pair[0] + "\n")
    antigen.write(">\n")
    antigen.write(pair[1] + "\n")
    lables.write(str(pair[3]) + "\n")
