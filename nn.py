from initializer import *


class SequentialModel:
    def __init__(self, layers, optimizer=None, regularizer=None):
        self.layers = layers
        self.optimizer = optimizer
        self.regularizer = regularizer

        self.size = len(layers)
        self.layer_dims = []
        self.loss = None
        self.costs = []
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


    def forward_prop(self, X):
        A = X
        for layer in self.layers:
            A = layer.layer_forward(A)
        return A


    def back_prop(self, dA):
        for layer in reversed(self.layers):
            dA = layer.layer_backward(dA)
        return dA


    def update_parameters(self, learning_rate):
        for layer in self.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db


    def compute_cost(self, AL, Y, loss):
        m = Y.shape[1]
        if loss == 'binary_crossentropy':
            cost = -1 / m * (np.sum(Y * np.log(AL)) + np.sum((1 - Y) * np.log(1 - AL)))
            cost = np.squeeze(cost)
        return cost


    def predict(self, X, Y):
        AL = self.forward_prop(X)
        prediction = (AL >= 0.5)
        accuracy = (np.sum(prediction * Y) + np.sum((1 - prediction) * (1 - Y))) / X.shape[1] * 100
        return prediction, accuracy

    def get_loss_deriv(self, AL, Y, loss):
        if loss == 'binary_crossentropy':
            dA = - (Y / AL - (1 - Y) / (1 - AL))
        return dA

    def fit(self, X, Y, loss='binary_crossentropy', optimizer=None, num_iterations=1000, learning_rate=0.075, print_status=True):
        self.initialize_parameters(input_dim=X.shape[0])

        for i in range(num_iterations):
            AL= self.forward_prop(X)
            cost = self.compute_cost(AL, Y, loss)
            loss_deriv = self.get_loss_deriv(AL, Y, loss)
            dA = self.back_prop(loss_deriv)
            self.update_parameters(learning_rate)

            if i % 100 == 0:
                self.costs.append(cost)
                if print_status:
                    print(f"{i}th iteration cost: {cost}")






