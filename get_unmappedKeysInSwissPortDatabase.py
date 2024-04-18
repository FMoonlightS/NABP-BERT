"""
@author: Fatma S. Ahmed
@brief: get unmapped Keys In Swiss-Port Database.
"""

import pickle

with open('get_unmappedKeysInSwissPortDatabase.txt','w') as f:
    # load the binnarized dataset:
    with open("dataAfterPreProcessing/SwissDataBase/SwissDataBase.pickle","rb") as input_:
        database_Swiss=pickle.load(input_)
    print("size of Swissport database : " + str(len(database_Swiss)) + "\n")
    f.write("size of Swissport database : " + str(len(database_Swiss)) + "\n")

    with open('dataAfterPreProcessing/PPI_Dataset/unique_protein_set.pickle','rb') as input_:
        database_unique=pickle.load(input_)
    print("size of unique_protein_set database : " + str(len(database_unique)) + "\n")
    f.write("size of unique_protein_set database : " + str(len(database_unique)) + "\n")

    # Checking the validitiy of the keys:
    print("Checking the validitiy of the keys:\n")
    f.write("Checking the validitiy of the keys:\n")
    curratedKeys=set()
    unmappedKeys=set()
    SwissKeys=database_Swiss.keys()
    for key in database_unique:
        if key in SwissKeys:
            curratedKeys.add(key)
        else:
            unmappedKeys.add(key)

    print("Number of unique keys: "+str(len(database_unique)) + "\n")
    print("Number of mapped keys: "+str(len(curratedKeys))+ "\n")
    print("Number of unmapped keys: "+str(len(unmappedKeys))+ "\n")
    print("Success Percentage: "+str(
            (len(curratedKeys)/len(database_unique))*100)+"%" + "\n")

    f.write("Number of unique keys: "+str(len(database_unique)) + "\n")
    f.write("Number of mapped keys: "+str(len(curratedKeys))+ "\n")
    f.write("Number of unmapped keys: "+str(len(unmappedKeys))+ "\n")
    f.write("Success Percentage: "+str(
            (len(curratedKeys)/len(database_unique))*100)+"%" + "\n")

    with open("dataAfterPreProcessing/SwissDataBase/unmappedKeysInSwissPortDatabase.pickle","wb") as output_:
        pickle.dump(unmappedKeys,output_)
