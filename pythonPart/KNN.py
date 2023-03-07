import itertools
import numpy as np
import multiprocessing
import random
from math import exp, sqrt
from dtwn import DTW
from kernel import calculateHelperMatrix, calculateGramMatrix, DTWGaussKernel
from sklearn import svm
from joblib import Parallel, delayed
from bisect import bisect_left


class KNN:
    def __init__(self, k = 6):
        self.k = int(k)

    def train(self, train_set):
        self.train_set = list(train_set)

        self.helper_matrix = np.sort(calculateHelperMatrix(self.train_set), axis = 1)
        self.distances = sorted(list(self.helper_matrix[:, self.k + 1]))

    def classify(self, record):
        num_cores = multiprocessing.cpu_count()
        distances_for_record = Parallel(n_jobs = num_cores)(delayed(DTW)(record, train_record) for train_record in self.train_set)
        distances_for_record.sort()
        distance = distances_for_record[self.k]

        result = 1.0 - bisect_left(self.distances, distance) / float(len(self.distances))
        return result

