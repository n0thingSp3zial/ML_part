import multiprocessing, itertools
import numpy as np
from dtaidistance import dtw_ndim
from joblib import Parallel, delayed
from math import exp

def calculateHelperMatrix(train_data):
    num_cores = multiprocessing.cpu_count()
    distance_matrix_parallel = Parallel(n_jobs = num_cores)(delayed(dtw_ndim.distance)(*pair) for pair in itertools.product(train_data, repeat = 2))
    distance_matrix_parallel = np.array(distance_matrix_parallel).reshape((len(train_data), len(train_data)))

    return distance_matrix_parallel

def calculateGramMatrix(helper_matrix):

    gram_matrix = []
    max_distance = np.max(helper_matrix)

    for row in helper_matrix:
        new_row = []
        for elem in row:
            new_elem = exp(-1.0 * elem / (max_distance ** 2))
            new_row.append(new_elem)
        gram_matrix.append(new_row)

    return gram_matrix, max_distance

def DTWGaussKernel(record1, record2, sigma):
    return exp(-1.0 * dtw_ndim.distance(record1, record2) / (sigma ** 2))
