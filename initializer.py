import numpy as np


class Initializer:
    # zero_init is only for logistic regression
    @staticmethod
    def one(dim_prev, dim_curr):
        W = np.ones(dim_curr, dim_prev)
        b = np.ones((dim_curr, 1))
        return W, b

    @staticmethod
    def zero(dim_prev, dim_curr):
        W = np.zeros(dim_curr, dim_prev)
        b = np.zeros((dim_curr, 1))
        return W, b

    @staticmethod
    def random(dim_prev, dim_curr):
        W = np.random.randn(dim_curr, dim_prev) * 0.01
        b = np.zeros((dim_curr, 1))
        return W, b

    # sets variance of weight as 2 / n (where n == number of nodes in previous layer)
    @staticmethod
    def he(dim_prev, dim_curr):
        W = np.random.randn(dim_curr, dim_prev) * np.sqrt(2 / dim_prev)
        b = np.zeros((dim_curr, 1))
        return W, b