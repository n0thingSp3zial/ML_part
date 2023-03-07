import os, json, fnmatch
from uWave import *
import numpy as np

############################################################
def createFoldersForUser(username, modelname):
    userFolder = os.path.join(DATA_DIR, username)
    modelFolder = os.path.join(userFolder, modelname)
    
    trainDataFolder = os.path.join(modelFolder, "train")
    testDataFolder = os.path.join(modelFolder, "test")
    attackDataFolder = os.path.join(modelFolder, "attack")
    
    # Create train data folder if it does not exist
    if not os.path.exists(trainDataFolder):
        os.makedirs(trainDataFolder)
        
    # Create test data folder if it does not exist
    if not os.path.exists(testDataFolder):
        os.makedirs(testDataFolder)
        
    # Create attack data folder if it does not exist
    if not os.path.exists(attackDataFolder):
        os.makedirs(attackDataFolder)
        
def writeDataToFile(data, filepath):
    with open(filepath, 'w') as out_file:
        out_file.write(json.dumps(json.loads(data), sort_keys = True, indent = 4, separators = (',', ': ')))
        
def getQuantizedData(folder):
    dataList = []
    for file in os.listdir(folder):
        if fnmatch.fnmatch(file, '*.txt'):
            with open(os.path.join(folder, file), 'r') as trainFileName:
                trainFileString = trainFileName.read()
                trainFile = json.loads(trainFileString)
                trainFileData = trainFile["params"]["data"]
                
                trainAccelerationData = [x["acceleration"] for x in trainFileData]
                trainFrequency = trainFile["params"]["frequency"]
                
                dataList.append(quantizeData(trainAccelerationData, trainFrequency))
        if fnmatch.fnmatch(file, '*.json'):
            with open(os.path.join(folder, file), 'r') as gesture_file:
                gesture_in_json = gesture_file.read()
                time_points = json.loads(gesture_in_json)

                x_acc = np.array([d['x_acc'] for d in time_points])
                y_acc = np.array([d['y_acc'] for d in time_points])
                z_acc = np.array([d['z_acc'] for d in time_points])
                
                a_rot = np.array([d['a_rot'] for d in time_points])
                b_rot = np.array([d['b_rot'] for d in time_points])
                g_rot = np.array([d['g_rot'] for d in time_points])

                train_freq = 100
                train_acc_data = np.stack([x_acc, y_acc, z_acc]).T.tolist()
                
                dataList.append(quantizeData(train_acc_data, train_freq))
                
    return dataList    
############################################################
