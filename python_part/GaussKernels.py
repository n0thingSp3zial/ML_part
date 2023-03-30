import multiprocessing
from joblib import Parallel, delayed
import itertools
import numpy as np
from math import exp, sqrt
from dtwn import DTW
from scipy.spatial.distance import pdist


class GaussKernel:

    def __init__(self, classifier_type):
        self.classifier_type = classifier_type

    metric = lambda self, x, y: pdist([x, y]) ** 2
    max_distance = None

    def calculateGramMatrix(self, train_data):
        squared_distances = [self.metric(*pair) for pair in itertools.product(train_data, repeat=2)]
        squared_distances = np.array(squared_distances).reshape((len(train_data), len(train_data)))
        gram_matrix = []
        self.max_distance = np.max(squared_distances)
        for row in squared_distances:
            new_row = []
            for elem in row:
                new_elem = exp(-1.0 * elem / (self.max_distance ** 2))
                new_row.append(new_elem)
            gram_matrix.append(new_row)
        return gram_matrix

    def calculate(self, x, y, sigma=None):
        if sigma == None:
            sigma = self.max_distance
        if self.classifier_type == 'SVM':
            sigma = sqrt(sigma)
        return exp(-1.0 * self.metric(x, y) / (sigma ** 2))


class DTWGaussKernel(GaussKernel):
    
    metric = lambda self, x, y: DTW(x, y)

    def calculateGramMatrix(self, train_data):
        num_cores = multiprocessing.cpu_count()
        squared_distances = Parallel(n_jobs=num_cores)(delayed(DTW)(*pair) for pair in itertools.product(train_data, repeat=2))
        squared_distances = np.array(squared_distances).reshape((len(train_data), len(train_data)))

        gram_matrix = []
        self.max_distance = np.max(squared_distances)
        for row in squared_distances:
            new_row = []
            for elem in row:
                new_elem = exp(-1.0 * elem / (self.max_distance ** 2))
                new_row.append(new_elem)
            gram_matrix.append(new_row)
        return gram_matrix
