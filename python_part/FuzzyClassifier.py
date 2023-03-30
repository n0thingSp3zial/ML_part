import itertools
import numpy as np
import random


class FuzzyClassifier:
    def __init__(self, kernel, m=2, max_iterations=1000, eps=1e-8, k=0.8):
        self.u_l = None
        self.nu = None
        self.gram_matrix = None

        self.kernel = kernel
        self.m = m
        self.max_iterations = max_iterations
        self.eps = eps
        self.k = k

    def train(self, train_set):
        self.train_set = list(train_set)

        # Step 0
        self.u_l = [random.random() for _ in range(len(self.train_set))]
        # self.nu = 0.5
        self.gram_matrix = self.kernel.calculateGramMatrix(self.train_set)

        current_iteration = 0
        while (True):
            current_iteration += 1

            a_denom = sum(x ** self.m for x in self.u_l)
            a = [(x ** self.m) / a_denom for x in self.u_l]

            distances_elem_1 = sum([a[i] * a[j] * self.gram_matrix[i][j] for (i, j) \
                                    in itertools.product(range(len(self.train_set)), repeat=2)])
            distances = [distances_elem_1 + self.gram_matrix[n][n] - \
                         2 * sum([a[j] * self.gram_matrix[j][n] for j in range(len(self.train_set))]) \
                         for n in range(len(self.train_set))]

            self.nu = sorted(distances, reverse=True)[int((1.0 - self.k) * len(self.train_set))]

            u_l_1 = [1.0 / (1 + (distances[n] / self.nu) ** (1.0 / (self.m - 1))) for n in range(len(self.train_set))]

            if (np.linalg.norm(np.array(self.u_l) - np.array(u_l_1)) > self.eps) and (
                        current_iteration < self.max_iterations):
                self.u_l = u_l_1
                continue
            else:
                break

    def classify(self, record):
        elem1_numer = sum([(self.u_l[i] ** self.m) * (self.u_l[j] ** self.m) * self.gram_matrix[i][j] \
                           for (i, j) in itertools.product(range(len(self.train_set)), repeat=2)])
        elem1_denom = self.nu * (sum([u ** self.m for u in self.u_l]) ** 2)
        elem1 = elem1_numer / elem1_denom

        elem2_numer = sum([(self.u_l[i] ** self.m) * (self.kernel.calculate(record, self.train_set[i])) \
                           for i in range(len(self.train_set))])
        elem2_denom = self.nu * (sum([u ** self.m for u in self.u_l]))
        elem2 = elem2_numer / elem2_denom

        elem3 = self.kernel.calculate(record, record) / self.nu

        result = 1.0 / (1.0 + (elem1 - 2.0 * elem2 + elem3) ** (1.0 / (self.m - 1)))

        return result