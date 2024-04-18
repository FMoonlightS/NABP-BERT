"""
@author: Fatma S. Ahmed
@brief: prepare PPI Sequences.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## loading the modules:
import os 
import pandas as pd
import pickle 

dir='./data/PPI_data'
pair_lst=set()

with open('CleanHINTdatabase.txt','w') as f:

    print("I am loading data from the directory :"+dir+"\n")
    f.write("I am loading data from the directory :"+dir+"\n")
    files=os.listdir(dir+"/")
    for file in files:
        print("file:"+dir+ "/" + file + "\n")
        f.write("file:"+dir+ "/" + file + "\n")

        dum_table=pd.read_table(dir+"/"+file)

        print(" len of file:"+dir+ "/" + file + " : " + str(len(dum_table)) + "\n")
        f.write(" len of file:"+dir+ "/" + file + " : " + str(len(dum_table)) + "\n")

        for a_tuple in dum_table.itertuples():
            pair_lst.add((a_tuple[1],a_tuple[2]))
        print("current PPI database size is : {}\n".format(len(pair_lst)))
        f.write("current PPI database size is : {}\n".format(len(pair_lst)))

    ## save the results as a pickle object on the results directory
    with open('dataAfterPreProcessing/PPI_Dataset/set_pairs.pickle','wb') as binary_writer:
        pickle.dump(pair_lst,binary_writer)

    ## getting the set of unique proteins
    unique_proteins=set()
    for protein_pair in pair_lst:
        unique_proteins.add(protein_pair[0])
        unique_proteins.add(protein_pair[1])
    print("number of unique proteins is {}\n".format(len(unique_proteins)))
    f.write("number of unique proteins is {}\n".format(len(unique_proteins)))

    ## save the results as a pickle object on the results directory
    with open('dataAfterPreProcessing/PPI_Dataset/unique_protein_set.pickle', 'wb') as binary_writer:
        pickle.dump(unique_proteins,binary_writer)
