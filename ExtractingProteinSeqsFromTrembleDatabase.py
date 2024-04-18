"""
@author: Fatma S. Ahmed
@brief: Extracting Protein Seqs From Tremble Database.
"""

import pickle
with open('ExtractingProteinSeqsFromTrembleDatabase.txt','w') as f:

    with open("dataAfterPreProcessing/SwissDataBase/unmappedKeysInSwissPortDatabase.pickle","rb") as input_:
        unmappedKeysInSwissPortDatabase=pickle.load(input_)

    curratedKeys=set()
    unmappedKeys=set()
    curratedPPIDataBaseFromTremble=dict()

    for i in range(1,52):
        with open("dataAfterPreProcessing/TrembleDataBase/TrembleDataBase_"+str(i)+".pickle", "rb") as input_:
            TrembleDataBase = pickle.load(input_)

        TrembleKeys=TrembleDataBase.keys()
        for key in unmappedKeysInSwissPortDatabase:
            if key in TrembleKeys:
                curratedKeys.add(key)
                curratedPPIDataBaseFromTremble[key]=TrembleDataBase[key]
            else:
                unmappedKeys.add(key)

        print("Number of unmapped Keys Before Search in Tremble Database "+str(i)+": "+str(len(unmappedKeysInSwissPortDatabase)) + "\n")
        print("Number of mapped keys in Tremble Database "+str(i)+": "+str(len(curratedKeys))+ "\n")
        print("Number of unmapped keys after Search in Tremble Database "+str(i)+": "+str(len(unmappedKeys))+ "\n")
        print("Success Percentage: "+str(
                (len(curratedKeys)/len(unmappedKeysInSwissPortDatabase))*100)+"%" + "\n")
        print("=============================================================================\n")
        f.write("Number of unmapped Keys Before Search in Tremble Database "+str(i)+": "+str(len(unmappedKeysInSwissPortDatabase)) + "\n")
        f.write("Number of mapped keys in Tremble Database "+str(i)+": "+str(len(curratedKeys))+ "\n")
        f.write("Number of unmapped keys after Search in Tremble Database "+str(i)+": "+str(len(unmappedKeys))+ "\n")
        f.write("Success Percentage: "+str(
                (len(curratedKeys)/len(unmappedKeysInSwissPortDatabase))*100)+"%" + "\n")
        f.write("=============================================================================\n")

        unmappedKeysInSwissPortDatabase = unmappedKeys.copy()
        unmappedKeys.clear()

    with open("dataAfterPreProcessing/TrembleDataBase/curratedKeys.pickle","wb") as output_:
        pickle.dump(curratedKeys,output_)

    with open("dataAfterPreProcessing/TrembleDataBase/unmappedKeys.pickle","wb") as output_:
        pickle.dump(unmappedKeysInSwissPortDatabase,output_)

    with open("dataAfterPreProcessing/TrembleDataBase/curratedPPIDataBaseFromTremble.pickle","wb") as output_:
        pickle.dump(curratedPPIDataBaseFromTremble,output_)
