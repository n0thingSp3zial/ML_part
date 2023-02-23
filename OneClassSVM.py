import itertools
import numpy as np
import random
from math import exp, sqrt
from kernel import calculateHelperMatrix, calculateGramMatrix, DTWGaussKernel
from sklearn import svm


class OneClassSVM:
    def __init__(self, nu):
        self.nu = nu
        self.sigma = None
        self.gram_matrix = None

    def train(self, train_set):
        self.train_set = list(train_set)

        #self.nu = 2.0 / len(self.train_set)
        self.gram_matrix, self.max_distance = calculateGramMatrix(calculateHelperMatrix(self.train_set))
        self.sigma = sqrt(self.max_distance)

        self.clf = svm.OneClassSVM(nu = self.nu, kernel = 'precomputed')
        self.clf.fit(self.gram_matrix)
        self.support_ = self.clf.support_
        self.coefficients = self.clf.dual_coef_.tolist()[0]
        self.rho = self.clf.intercept_[0]


    def classify(self, record):
        result = sum( [ coef * DTWGaussKernel(self.train_set[num], record, self.sigma)
                        for (num, coef) in zip(self.support_, self.coefficients) ] )
        return result + self.rho
