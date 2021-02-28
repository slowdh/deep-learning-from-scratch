import numpy as np


class Optimizer:
    pass

class SGD(Optimizer):
    def update_parameters(self, layer, grads):
        return grads

    def step(self, layer, learning_rate):
        W_step = learning_rate * layer.dW
        b_step = learning_rate * layer.db
        return W_step, b_step


class Momentum(Optimizer):
    def __init__(self, beta=0.9):
        # layer name -> weight average
        self.v_average = {}
        self.beta = beta

    def update_parameters(self, layer, grads):
        # zero initialize for velocity of each parameters
        if layer not in self.v_average:
            Vw = np.zeros(layer.W.shape)
            Vb = np.zeros(layer.b.shape)
            self.v_average[layer] = (Vw, Vb)

        # update velocity
        dW, db = grads
        Vw, Vb = self.v_average[layer]
        Vw = self.beta * Vw + (1 - self.beta) * dW
        Vb = self.beta * Vb + (1 - self.beta) * db
        self.v_average[layer] = (Vw, Vb)
        return Vw, Vb

    def step(self, layer, learning_rate):
        Vw, Vb = self.v_average[layer]
        W_step = learning_rate * Vw
        b_step = learning_rate * Vb
        return W_step, b_step


class RMSprop(Optimizer):
    def __init__(self, beta, epsilon=1e-8):
        self.s_average = {}
        self.beta = beta
        self.epsilon = epsilon

    def update_parameters(self, layer, grads):
        # zero initialize for velocity of each parameters
        if layer not in self.s_average:
            Sw = np.zeros(layer.W.shape)
            Sb = np.zeros(layer.b.shape)
            self.s_average[layer] = (Sw, Sb)

        # update velocity
        dW, db = grads
        Sw, Sb = self.s_average[layer]
        Sw = beta * Sw + (1 - beta) * (dW ** 2)
        Sb = beta * Sb + (1 - beta) * (db ** 2)
        self.s_average[layer] = (Sw, Sb)

    def step(self, layer, learning_rate):
        Sw, Sb = self.s_average[layer]
        dW = layer.dW
        db = layer.db
        W_step = learning_rate * dW / (np.sqrt(Sw) + self.epsilon)
        b_step = learning_rate * db / (np.sqrt(Sb) + self.epsilon)
        return W_step, b_step

class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.averages = {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def update_parameters(self, layer, grads):
        # zero initialize for velocity of each parameters
        if layer not in self.averages:
            Vw = np.zeros(layer.W.shape)
            Vb = np.zeros(layer.b.shape)
            Sw = np.zeros(layer.W.shape)
            Sb = np.zeros(layer.b.shape)
            self.averages[layer] = (Vw, Vb, Sw, Sb)

        # update velocity
        dW, db = grads
        Vw, Vb, Sw, Sb = self.averages[layer]
        self.averages[layer] = (Vw, Vb, Sw, Sb)
        self.t += 1

    def step(self, layer, learning_rate):
        Vw, Vb, Sw, Sb = self.averages[layer]
        # bias correction
        Vw_corrected = Vw / (1 - self.beta1 ** self.t)
        Vb_corrected = Vb/ (1 - self.beta1 ** self.t)
        Sw_corrected = Sw / (1 - self.beta2 ** self.t)
        Sb_corredted = Sb / (1 - self.beta2 ** self.t)
        W_step = learning_rate * Vw_corrected / (np.sqrt(Sw_corrected) + self.epsilon)
        b_step = learning_rate * Vb_corrected / (np.sqrt(Sb_corredted) + self.epsilon)
        return W_step, b_step