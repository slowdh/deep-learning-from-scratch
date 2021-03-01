import numpy as np


class Regularizer:
    def forward(self, A_prev, W):
        A = A_prev
        reg_cost = 0
        return A, reg_cost

    def backward(self, dA):
        dW, db = 0, 0
        dA_prev = dA
        return (dW, db, dA_prev)


class L2(Regularizer):
    def __init__(self, lambd=0.1):
        self.lambd = lambd

    def forward(self, A_prev, m, W):
        reg_cost = 1 / m * (self.lambd / 2) * np.sum(W ** 2)
        return A_prev, reg_cost

    def backward(self, dA, m, W):
        dreg = self.lambd / m * W
        return dA, dreg


class Dropout(Regularizer):
    def __init__(self, keep_prob=0.7):
        self.keep_prob = keep_prob
        self.mask = 1

    def forward(self, A, m, W):
        self.mask = (np.random.rand(A.shape[0], A.shape[1]) < self.keep_prob).astype(int)
        A_masked = self.mask * A / self.keep_prob
        return A_masked, 0

    def backward(self, dA, m, W):
        dA_masked = dA * self.mask / self.keep_prob
        return dA_masked, 0
