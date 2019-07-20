import numpy as np

class Regularization(object):

    @staticmethod
    def L1(weights, lmbda):
        return lmbda * np.nan_to_num(np.sign(weights))

    @staticmethod
    def L2(weights, lmbda):
        return lmbda * weights

    @staticmethod
    def none(weights, lmbda):
        return 0
