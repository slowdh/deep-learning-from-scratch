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


    def layer_forward(self, A_prev):
        Z = np.dot(self.W, A_prev) + self.b

        if self.activation == 'sigmoid':
            A = sigmoid(Z)
        elif self.activation == 'relu':
            A = relu(Z)

        self.cache = (A_prev, Z)
        return A


    def layer_backward(self, dA):
        A_prev, Z = self.cache
        m = dA.shape[1]

        if self.activation == 'sigmoid':
            dZ = dA * sigmoid_derivative(Z)
        elif self.activation == 'relu':
            dZ = dA * relu_derivative(Z)

        self.dW = 1 / m * np.dot(dZ, A_prev.T)
        self.db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)
        return dA_prev
