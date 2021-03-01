import numpy as np
from initializer import *


class SequentialModel:
    def __init__(self, layers):
        self.layers = layers

        self.size = len(layers)
        self.layer_dims = []
        self.loss = None
        self.metrics = None
        self.parameter = None


    # only work for Dense layer now
    # todo: What if Conv layer comes?
    def initialize_parameters(self, input_dim):
        dim_prev = input_dim
        for layer in self.layers:
            shape = (layer.dim, dim_prev)
            if layer.init_method == 'zero':
                W, b = zero_init(shape)
            elif layer.init_method == 'random':
                W, b = random_init(shape)
            elif layer.init_method == 'he':
                W, b = he_init(shape)

            dim_prev = layer.dim
            layer.W = W
            layer.b = b


    def forward_prop(self, X, is_training=True):
        A = X
        reg_cost_sum = 0
        for layer in self.layers:
            Z, A, reg_cost = layer.layer_forward(A, is_training)
            reg_cost_sum += reg_cost
        return Z, A, reg_cost_sum


    def back_prop(self, dA, optimizer):
        for layer in reversed(self.layers):
            dA = layer.layer_backward(dA)
            optimizer.update_parameters(layer, (layer.dW, layer.db))
        return dA


    def update_parameters(self, learning_rate, optimizer):
        for layer in self.layers:
            W_step, b_step = optimizer.step(layer, learning_rate)
            layer.W -= W_step
            layer.b -= b_step


    def compute_cost(self, Z, AL, Y, reg_cost, loss):
        m = Y.shape[1]
        if loss == 'binary_crossentropy':
            cost = 1 / m * (np.sum(np.maximum(Z, 0)) - np.sum(Z * Y) + np.sum(np.log(1 + np.exp(-np.abs(Z)))))
            cost += reg_cost
            cost = np.squeeze(cost)
        return cost


    def predict(self, X, Y):
        Z, AL, _ = self.forward_prop(X, is_training=False)
        prediction = AL >= 0.5
        accuracy = (np.sum(prediction * Y) + np.sum((1 - prediction) * (1 - Y))) / X.shape[1] * 100
        return prediction, accuracy


    def get_loss_deriv(self, AL, Y, loss):
        if loss == 'binary_crossentropy':
            dA = - (Y / AL - (1 - Y) / (1 - AL))
        return dA


    def get_mini_batches(self, X, Y, batch_size):
        if batch_size is None:
            return [(X, Y)]

        m = X.shape[1]
        permutation = np.random.permutation(m)
        mini_batches = []
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]

        num_batch = m // batch_size
        remainder = m % batch_size
        if remainder != 0:
            num_batch += 1

        for k in range(0, num_batch):
            mini_batch_X = X_shuffled[:, k * batch_size:(k + 1) * batch_size]
            mini_batch_Y = Y_shuffled[:, k * batch_size:(k + 1) * batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches


    def fit(self, X_train, Y_train, loss='binary_crossentropy', optimizer=None, batch_size=None, epochs=1000, learning_rate=0.075, print_status=True, print_freq=100):
        self.initialize_parameters(input_dim=X_train.shape[0])

        costs = []
        for i in range(epochs):
            for mini_batch in self.get_mini_batches(X_train, Y_train, batch_size):
                X, Y = mini_batch
                Z, AL, reg_cost = self.forward_prop(X)
                cost = self.compute_cost(Z, AL, Y, reg_cost, loss)
                loss_deriv = self.get_loss_deriv(AL, Y, loss)
                self.back_prop(loss_deriv, optimizer)
                self.update_parameters(learning_rate, optimizer)

            if (i + 1) % print_freq == 0:
                costs.append(cost)
                if print_status:
                    print(f"{i + 1}th epoch cost: {cost}")
        print(f"Final cost: {cost}")
        return costs


# Todo: Training, Dev set -> get seperate cost, accuracy