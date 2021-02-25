import numpy as np
from nn import SequentialModel
from layers import DenseLayer
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


# make model
A = SequentialModel(layers=[
    DenseLayer(dim=7, activation='relu'),
    DenseLayer(dim=1, activation='sigmoid')
])

A.fit(x_train, y_train)
pred, acc = A.predict(x_test, y_test)
print(acc)
