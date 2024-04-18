"""
@author: Fatma S. Ahmed
@brief: create Swiss-port and Tremble Databases.
"""

import os
import pandas as pd
import pickle

def FASTA2Dic_SwissDataBase(infile):
    dataBase = dict()
    seq = ''
    key = ''
    i = 0
    with open(infile,'r') as f:
        # lines = f.readlines()
        for line in f:
            if line[0] == ">":
                if key != '':
                    dataBase[key] = seq
                    i += 1
                    # print(str(i) + '- ' + key + ' : '+ seq)
                    print(str(i) + ' - ' + key + '\n')
                    seq = ''
                    key = ''
                indexes = [i for i, letter in enumerate(line) if letter == '|']
                start = indexes[0]
                end = indexes[1]
                key = line[start + 1:end]

            else:
                seq += line.strip()
    f.close()
    return dataBase

def FASTA2Dic_TrembleDatabase(infile):
    dataBase = dict()
    seq = ''
    key = ''
    i = 0
    j = 0
    with open(infile,'r') as f:
        # lines = f.readlines()
        for line in f:
            if line[0] == ">":
                if key != '':
                    dataBase[key] = seq
                    i += 1
                    # print(str(i) + '- ' + key + ' : '+ seq)
                    print(str(i) + ' - ' + key + '\n')
                    seq = ''
                    key = ''
                    if i % 5000000 == 0:
                        j += 1
                        with open('dataAfterPreProcessing/TrembleDataBase/TrembleDataBase_' + str(j) + '.pickle', 'wb') as binary_writer:
                            pickle.dump(dataBase, binary_writer)
                        dataBase.clear()
                indexes = [i for i, letter in enumerate(line) if letter == '|']
                start = indexes[0]
                end = indexes[1]
                key = line[start + 1:end]

            else:
                seq += line.strip()
        else:
            j += 1
            with open('dataAfterPreProcessing/TrembleDataBase/TrembleDataBase_' + str(j) + '.pickle' , 'wb') as binary_writer:
                pickle.dump(dataBase, binary_writer)
            dataBase.clear()

    f.close()
    # return dataBase


infile = 'data/uniprot_sprot.fasta/uniprot_sprot.fasta'
database = FASTA2Dic_SwissDataBase(infile)
with open('dataAfterPreProcessing/SwissDataBase/SwissDataBase.pickle', 'wb') as binary_writer:
    pickle.dump(database,binary_writer)

infile = 'data/uniprot_trembl.fasta'
FASTA2Dic_TrembleDatabase(infile)

print('done')
