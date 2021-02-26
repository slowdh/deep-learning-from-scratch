import numpy as np
from activations import *

class DenseLayer:
    def __init__(self, dim, activation, init_method='random', regularizer=None):
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
        reg_cost = 0
        if is_training and self.regularizer is not None:
            reg, rate = self.regularizer
            if reg == 'l2':
                reg_cost += 1 / m * (rate / 2) * np.sum(self.W ** 2)
            elif reg == 'dropout':
                self.mask = (np.random.rand(Z.shape[0], Z.shape[1]) < rate).astype(int)
        else:
            self.mask = 1
            rate = 1

        # forward prop
        if self.activation == 'sigmoid':
            A = sigmoid(Z)
        elif self.activation == 'relu':
            A = relu(Z)

        A = A * self.mask / rate
        self.cache = (A_prev, Z)
        return A, reg_cost


    def layer_backward(self, dA):
        A_prev, Z = self.cache
        m = dA.shape[1]

        dreg = 0
        if self.regularizer is not None:
            reg, rate = self.regularizer
            if reg == 'l2':
                dreg = rate / m * self.W
        else:
            rate = 1
            self.mask = 1

        dA = dA * self.mask / rate
        if self.activation == 'sigmoid':
            dZ = dA * sigmoid_derivative(Z)
        elif self.activation == 'relu':
            dZ = dA * relu_derivative(Z)

        self.dW = 1 / m * np.dot(dZ, A_prev.T) + dreg
        self.db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)
        return dA_prev
