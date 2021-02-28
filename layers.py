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
        self.mask = 1


    def layer_forward(self, A_prev, is_training):
        m = A_prev.shape[1]
        Z = np.dot(self.W, A_prev) + self.b

        # about regularization
        keep_prob = 1
        reg_cost = 0
        if is_training and self.regularizer is not None:
            reg, rate = self.regularizer
            if reg == 'l2':
                lambd = rate
                reg_cost += 1 / m * (rate / 2) * np.sum(self.W ** 2)
            elif reg == 'dropout':
                keep_prob = rate
                self.mask = (np.random.rand(Z.shape[0], Z.shape[1]) < rate).astype(int)

        # forward prop
        if self.activation == 'sigmoid':
            A = sigmoid(Z)
        elif self.activation == 'relu':
            A = relu(Z)

        if is_training:
            A = A * self.mask / keep_prob
        self.cache = (A_prev, Z)
        return Z, A, reg_cost


    def layer_backward(self, dA):
        A_prev, Z = self.cache
        m = dA.shape[1]

        dreg = 0
        keep_prob = 1
        if self.regularizer is not None:
            reg, rate = self.regularizer
            if reg == 'l2':
                lambd = rate
                dreg = lambd / m * self.W
            elif reg == 'dropout':
                keep_prob = rate

        dA = dA * self.mask / keep_prob
        if self.activation == 'sigmoid':
            dZ = dA * sigmoid_derivative(Z)
        elif self.activation == 'relu':
            dZ = dA * relu_derivative(Z)

        self.dW = 1 / m * np.dot(dZ, A_prev.T) + dreg
        self.db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)
        return dA_prev