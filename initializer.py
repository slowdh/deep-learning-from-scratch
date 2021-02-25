import numpy as np


# zero_init is only for logistic regression
def zero_init(dim):
    curr, prev = dim
    W = np.zeros(curr, prev)
    b = np.zeros((curr, 1))
    return W, b

def random_init(dim):
    curr, prev = dim
    W = np.random.randn(curr, prev) * 0.01
    b = np.zeros((curr, 1))
    return W, b

# sets variance of weight as 2 / n (where n == number of nodes in previous layer)
def he_init(dim):
    curr, prev = dim
    W = np.random.randn(curr, prev) * np.sqrt(2 / prev)
    b = np.zeros((curr, 1))
    return W, b