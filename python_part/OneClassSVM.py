from sklearn import svm


class OneClassSVM:
    def __init__(self, kernel, nu):
        self.nu = nu
        self.gram_matrix = None
        self.kernel = kernel

    def train(self, train_set):
        self.train_set = list(train_set)

        #self.nu = 2.0 / len(self.train_set)
        self.gram_matrix = self.kernel.calculateGramMatrix(self.train_set)

        self.clf = svm.OneClassSVM(nu = self.nu, kernel = 'precomputed')
        self.clf.fit(self.gram_matrix)
        self.support_ = self.clf.support_
        self.coefficients = self.clf.dual_coef_.tolist()[0]
        self.rho = self.clf.intercept_[0]


    def classify(self, record):
        result = sum( [ coef * self.kernel.calculate(self.train_set[num], record)
                        for (num, coef) in zip(self.support_, self.coefficients) ] )
        return result + self.rho
