import numpy as np
from activations import *

class DenseLayer:
    def __init__(self, dim, activation, init_method='he', regularizer=None):
        self.dim = dim
        self.activation = activation
        self.init_method = init_method
        self.regularizer = regularizer

        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.cache = None


    def layer_forward(self, A_prev, is_training):
        m = A_prev.shape[1]
        Z = np.dot(self.W, A_prev) + self.b
        A = self.activation.forward(Z)
        reg_cost = 0

        if is_training and self.regularizer is not None:
            A, reg_cost = self.regularizer.forward(A, m, self.W)

        self.cache = (A_prev, Z)
        return Z, A, reg_cost


    def layer_backward(self, dA):
        A_prev, Z = self.cache
        m = dA.shape[1]
        dreg = 0

        if self.regularizer is not None:
            dA, dreg = self.regularizer.backward(dA, m, self.W)
        dZ = dA * self.activation.backward(Z)

        self.dW = 1 / m * np.dot(dZ, A_prev.T) + dreg
        self.db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)
        return dA_prev