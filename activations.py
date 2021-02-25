import numpy as np


def sigmoid(x):
    a = 1 / (1 + np.exp(-x))
    return a

def relu(x):
    a = np.maximum(0, x)
    return a

def sigmoid_derivative(x):
    s = sigmoid(x)
    d = s * (1 - s)
    return d

def relu_derivative(x):
    d = x.copy()
    d[d<=0] = 0
    d[d>0] = 1
    return d