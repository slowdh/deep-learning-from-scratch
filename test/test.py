import numpy as np
from models import SequentialModel
from layers import DenseLayer
from regularizer import *
import h5py

# get datasets
train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
x_train = train_dataset["train_set_x"][:]
y_train = train_dataset["train_set_y"][:].reshape((1,-1))

test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
x_test = test_dataset["test_set_x"][:]
y_test = test_dataset["test_set_y"][:].reshape((1,-1))

classes = test_dataset["list_classes"][:]

x_train = x_train.reshape(x_train.shape[0], -1).T / 255.
x_test = x_test.reshape((x_test.shape[0], -1)).T / 255.

print(f"x_train.shape: {x_train.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"x_test.shape: {x_test.shape}")
print(f"y_test.shape: {y_test.shape}")


reg = dropout(0.7)
# # simple 2 layer model (input_dim, 7, 1)
# print('Two layer model:')
# two_layer_model = SequentialModel(layers=[
#     DenseLayer(dim=7, activation='relu', regularizer=reg),
#     DenseLayer(dim=1, activation='sigmoid')
# ])
#
# two_layer_model.fit(x_train, y_train, num_iterations=1500)
# pred, acc = two_layer_model.predict(x_test, y_test)
# print(f'Two layer model\'s accuracy: {acc}')
# print()


print('Four layer model:')
# four layer model
four_layer_model = SequentialModel(layers=[
    DenseLayer(dim=20, activation='relu', init_method='he', regularizer=reg),
    DenseLayer(dim=7, activation='relu', init_method='he', regularizer=reg),
    DenseLayer(dim=5, activation='relu', init_method='he'),
    DenseLayer(dim=1, activation='sigmoid', init_method='he'),
])

four_layer_model.fit(x_train, y_train, learning_rate=0.005, num_iterations=1500)
pred, acc = four_layer_model.predict(x_test, y_test)
print(f'Four layer model\'s accuracy: {acc}')