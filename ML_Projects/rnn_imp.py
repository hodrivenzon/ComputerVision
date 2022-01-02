import numpy as np

np.random.seed(0)


# 7. Possible helper functions:
# a. Sigmoid
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# b. Sigmoid derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# c. Int to binary converter function
binary_dim = 8

largest_number = 2 ** binary_dim
to_binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)

# input variables
alpha = 0.1
iterations = 10000
x_dim = 2
h_dim = 48
y_dim = 1

# 1. Create a matrix for the weights of layer0,layer1 and the weights of the hidden layer
W_x = 2 * np.random.random((x_dim, h_dim)) - 1
W_y = 2 * np.random.random((h_dim, y_dim)) - 1
W_h = 2 * np.random.random((h_dim, h_dim)) - 1

# 2. We will run over 10,000 numbers, so the for loop should be over 10000 numbers
for j in range(iterations):

    # 3. For every iteration we will produce two random numbers and convert them to binary numbers
    # to use them as inputs and then take their addition as the “ground truth” of this calculation

    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number / 2)  # int version
    a = to_binary[a_int]  # binary encoding

    b_int = np.random.randint(largest_number / 2)  # int version
    b = to_binary[b_int]  # binary encoding

    # true answer
    c_int = a_int + b_int
    c = to_binary[c_int]

    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0

    layer_y_deltas = list()
    layer_h_values = list()
    layer_h_values.append(np.zeros(h_dim))

    # 4. Create a loop over the binary number to create the forward propagation and hence the errors
    for position in range(binary_dim):
        # generate input and output
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([c[binary_dim - position - 1]])
        prev_layer_h = layer_h_values[-1]

        # hidden layer (input ~+ prev_hidden)
        layer_h = sigmoid(X @ W_x + prev_layer_h @ W_h)
        # output layer (new binary representation)
        layer_y = sigmoid(layer_h @ W_y)
        layer_y = np.squeeze(layer_y)

        # did we miss?... if so, by how much?
        layer_y_error = layer_y - y
        layer_y_deltas.append(layer_y_error * sigmoid_output_to_derivative(layer_y))

        overallError += np.abs(layer_y_error[0])
        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_y)

        # store hidden layer so we can use it in the next timestep
        layer_h_values.append(layer_h.copy())

    future_layer_h_delta = np.zeros(h_dim)

    W_x_upd = np.zeros_like(W_x)
    W_y_upd = np.zeros_like(W_y)
    W_h_upd = np.zeros_like(W_h)

    # 5. Create another for loop over the binary number to calculate the deltas backpropagating to the weights
    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])
        layer_h = layer_h_values[-1 - position]
        prev_layer_h = layer_h_values[-2 - position]

        # error at output layer
        layer_y_delta = layer_y_deltas[-1 - position]
        # error at hidden layer
        layer_h_delta = (future_layer_h_delta @ W_h.T +
                         layer_y_delta @ W_y.T) * sigmoid_output_to_derivative(layer_h)

        # let's update all our weights so we can try again
        W_y_upd += np.atleast_2d(np.atleast_2d(layer_h).T.dot(layer_y_delta)).T
        W_h_upd += np.atleast_2d(prev_layer_h).T.dot(layer_h_delta)
        W_x_upd += X.T.dot(layer_h_delta)

        future_layer_h_delta = layer_h_delta

    # 6. Update the weights using Gradient Descent update rule
    W_x -= W_x_upd * alpha
    W_y -= W_y_upd * alpha
    W_h -= W_h_upd * alpha

    # print out progress
    if (j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")

