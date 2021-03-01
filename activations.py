import numpy as np

class Activation:
    pass

class Sigmoid(Activation):
    def forward(self, x):
        a = 1 / (1 + np.exp(-x))
        return a

    def backward(self, x):
        s = self.forward(x)
        d = s * (1 - s)
        return d

class Relu(Activation):
    def forward(self, x):
        a = np.maximum(0, x)
        return a

    def backward(self, x):
        d = x.copy()
        d[d <= 0] = 0
        d[d > 0] = 1
        return d