from math import sqrt

def quantizeData(accelerationsList, frequency):
    """Returns quantized accelerometer data
    accelerationsList - a list with accelerations
    frequency - frequency of sensor data
    """
    if (frequency not in set([10, 20, 40])):
        return accelerationsList
    
    quantizedData = []
    
    # Calculate sliding window parameters
    multiplier = 40 / frequency
    QUAN_WIN_SIZE, QUAN_MOV_STEP = multiplier * 2, multiplier * 1 
    
    window = QUAN_WIN_SIZE
    length = len(accelerationsList)
    
    # Calculating sliding window average
    for i in range(0, length, int(QUAN_MOV_STEP)):
        if (i + window > length):
            window = length - i
            
        summX, summY, summZ = 0, 0, 0
        
        for j in range(i, int(i + window)):
            summX += accelerationsList[j][0]
            summY += accelerationsList[j][1]
            summZ += accelerationsList[j][2]
            
        summX, summY, summZ = summX / window, summY / window, summZ / window
        
        quantizedData.append([summX, summY, summZ])
     
    '''   
    for i in xrange(0, len(quantizedData)):
        for l in xrange(0, 3):
            quantizedData[i][l] = int(quantizedData[i][l])
            if (quantizedData[i][l] > 10):
                if (quantizedData[i][l] > 20):
                    quantizedData[i][l] = 16
                else:
                    quantizedData[i][l] = 10 + (quantizedData[i][l] - 10) / 10 * 5
            elif (quantizedData[i][l] < -10):
                if (quantizedData[i][l] < -20):
                    quantizedData[i][l] = -16
                else:
                    quantizedData[i][l] = -10 + (quantizedData[i][l] + 10) / 10 * 5
    '''
    
    return quantizedData

def DTW(seqA, seqB, d = lambda x,y: sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)):
    """Returns a cost between two time sequences, got by dynamic time warping.
    seqA - a first time sequence
    seqB - a second time sequence
    d    - a function, which calculates distance between records
    """
    
    # create the cost matrix
    numRows, numCols = len(seqA), len(seqB)
    cost = [[0 for _ in range(numCols)] for _ in range(numRows)]
    
    # initialize the first row and column
    cost[0][0] = d(seqA[0], seqB[0])
    for i in range(1, numRows):
        cost[i][0] = cost[i-1][0] + d(seqA[i], seqB[0])
        
    for j in range(1, numCols):
        cost[0][j] = cost[0][j-1] + d(seqA[0], seqB[j])
        
    # fill in the rest of the matrix
    for i in range(1, numRows):
        for j in range(1, numCols):
            choices = cost[i-1][j], cost[i][j-1], cost[i-1][j-1]
            cost[i][j] = min(choices) + d(seqA[i], seqB[j])
            
    return cost[-1][-1]

def getNearestNeighbors(data, query, k):
    all_neighbors = []
        
    for neighbor in data:
        dist = DTW(query, neighbor)
        all_neighbors.append((neighbor, dist))

    all_neighbors.sort(key=lambda x: x[1])
        
    k_nearest_neighbors = []
    for x in xrange(k):
        k_nearest_neighbors.append(all_neighbors[x])
            
    return k_nearest_neighbors

def knn(trainList, k, test = None):
    """kNearestNeighbours algorithm
    trainList - a list with train records
    k - a parameter of quantity of nearest neighbours
    test - a test record
    
    If test record is given, it returns a tuple with distance for test record
    and list of distances for train set.
    If test record is omitted, it returns a tuple of one element - a list of
    distances for train set
    """
    
    distances = []
    for record in trainList:
        distances.append( getNearestNeighbors(trainList, record, k + 1)[-1][-1] )
    distances.sort()
    
    if test != None:
        trainDistance = getNearestNeighbors(trainList, test, k)[-1][-1]
        return (distances, trainDistance)
    else:
        return (distances, )