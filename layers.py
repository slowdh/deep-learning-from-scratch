import numpy as np
from activations import *
from initializer import *


class Layer:
    pass


class Dense(Layer):
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


    def initialize_parameters(self, dim_prev, dim_curr, method):
        if method == 'zero':
            self.W, self.b = Initializer.zero(dim_prev, dim_curr)
        elif method == 'random':
            self.W, self.b = Initializer.random(dim_prev, dim_curr)
        elif method == 'he':
            self.W, self.b = Initializer.he(dim_prev, dim_curr)


    def layer_forward(self, A_prev, is_training):
        if self.W is None:
            self.initialize_parameters(dim_prev=A_prev.shape[0], dim_curr=self.dim, method=self.init_method)

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


    def step(self, learning_rate, optimizer):
        # update parameters in layer
        W_step_size, b_step_size = optimizer.step(layer, learning_rate)
        layer.W -= W_step_size
        layer.b -= b_step_size


# WIP: Batch Normalization

class BatchNormalization(Layer):
    def __init__(self):
        self.gamma = None
        self.beta = None
        self.mean_avg = None
        self.var_avg = None
        self.dgamma = None
        self.dbeta = None
        self.cache = None

    def initialize_parameters(self, shape):
        self.gamma = np.ones((1, shape[1]))
        self.beta = np.zeros((1, shape[1]))

#     def layer_forward(self, Z, is_training):
#         # Z -> Z_tilda -> gamma * Z_tilda + beta
#         if self.gamma is None:
#             self.initialize_parameters(Z.shape)
#         if is_training:
#             pass
#
#
#     def layer_backward(self, dZ):
#         pass
#
#     def step(self):
#         pass