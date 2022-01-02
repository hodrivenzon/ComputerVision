import numpy as np
import matplotlib.pyplot as plt


# Calculate the transfer function of a neuron
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Calculate the derivative of a neuron output
def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    deriv = np.ones(len(x))
    deriv[x < 0] = 0
    return deriv


# one layer implementation

# initialize weights
np.random.seed(1)
weights = 2 * np.random.random((3, 1)) - 1
# weights *= 0.01
bias = 0
# weights = np.zeros(3,1) - 1

# Create data
x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([0, 0, 1, 1])

lr = 1
losses = []
for i in range(200):
    layer_1 = sigmoid(x @ weights + bias)
    error = layer_1.T - y
    l1_delta = error.T * sigmoid_derivative(layer_1)
    weights -= lr * np.dot(x.T, l1_delta)
    bias -= lr * l1_delta.sum()
    loss = 0.5 * np.sum(error ** 2)
    losses.append(loss)

print(layer_1)
plt.plot(losses)

## Two layers implementation

# initialize weights

w0 = 2 * np.random.random((3, 4)) - 1
w1 = 2 * np.random.random((4, 1)) - 1
# alpha = np.random.rand(1)
# beta = np.random.rand(1)
b0 = np.zeros(4)
b1 = 0
# Create data
x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([0, 0, 1, 1])

losses = []
for i in range(200):
    layer_1 = sigmoid(np.dot(x, w0) + b0)
    layer_2 = sigmoid(np.dot(layer_1, w1) + b1)

    l2_error = layer_2.T - y
    l2_delta = l2_error.T * sigmoid_derivative(layer_2)

    l1_error = np.dot(l2_delta, w1.T)
    l1_delta = l1_error * sigmoid_derivative(layer_1)

    w1 -= lr * np.dot(layer_1.T, l2_delta)
    b1 -= lr * l2_delta.sum()

    w0 -= lr * np.dot(x.T, l1_delta)
    b0 -= lr * l1_delta.sum(axis=0)

    loss = 0.5 * np.sum(l2_error ** 2)
    losses.append(loss)

    # plt.imshow(np.random.random((50, 50)))
    # plt.colorbar()
    # plt.show()

print(layer_2)
plt.plot(losses)