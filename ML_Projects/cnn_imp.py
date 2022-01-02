from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


class CNN:

    def __init__(self, num_filters, filter_size=3, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.lr = None
        self.filters = None
        self.bias = None
        self.weights = None
        self.bias2 = None

    def init_weights(self, image_size, num_labels):
        self.filters = np.random.random((self.filter_size, self.filter_size, self.num_filters)) - 0.5
        self.bias = np.random.random(self.num_filters) - 0.5
        conv_output_size = self.conv_output_size(image_size, self.filter_size, self.padding, self.stride)
        self.weights = np.random.random((self.num_filters * conv_output_size ** 2, num_labels)) - 0.5
        self.bias2 = np.random.random(num_labels) - 0.5

    def forward(self, batch):
        z1 = self.conv(batch, self.filters, self.padding, self.stride) + \
             self.bias[np.newaxis, np.newaxis, np.newaxis, :]
        # Relu: Replace all negative with zeros
        a1 = np.maximum(0, z1)
        # Flatten the matrices
        f1 = a1.reshape((a1.shape[0], -1))

        z2 = f1 @ self.weights + self.bias2
        a2 = self.softmax(z2)
        return z1, a1, f1, z2, a2

    def backward(self, batch, batches_y, cache):
        z1, a1, f1, z2, a2 = cache
        error = a2 - batches_y   #a2 is the output of the softmax
        error2 = (error @ self.weights.T).reshape(z1.shape) * np.heaviside(z1, 0)  # z1 is the convulsion output  ## np.heaviside if z1<0 return 0, f z1==0 return the second number (in this case 0), f z1>0 return 1
        self.weights -= self.lr * (f1.T @ error) # the output of relu reshaped
        self.bias2 -= self.lr * error.mean(axis=0)
        self.filters -= self.lr * self.conv(batch, error2, self.padding, self.stride).mean(axis=0)
        self.bias -= self.lr * error2.mean(axis=(0, 1, 2))

    def fit(self, x, y, lr=0.01, epochs=1000, batch_size=32, epoch_step=10):
        self.init_weights(x.shape[1], np.unique(y).shape[0])
        self.lr = lr
        onehot_y = self.onehot(y)
        for epoch in range(epochs):
            batches, batches_y = self.split(x, onehot_y, batch_size)
            for batch, batch_y in zip(batches, batches_y):
                cache = self.forward(batch)
                self.backward(batch, batch_y, cache)
            if epoch % epoch_step == 0:
                loss_now = self.cross_entropy_loss(self.forward(x)[-1], onehot_y).mean()
                print(f'epoch {epoch} loss {loss_now:.4f}')

    @staticmethod
    def split(x, y, batch_size):
        shuffle = np.random.permutation(y.shape[0])
        idx = np.arange(batch_size, x.shape[0], batch_size)
        batches = np.split(x[shuffle, :, :], idx)
        batch_y = np.split(y[shuffle, :], idx)
        return batches, batch_y

    @staticmethod
    def onehot(vec):
        r = np.zeros((vec.shape[0], np.unique(vec).shape[0]))
        r[range(len(vec)), vec] = 1
        return r

    @staticmethod
    def conv_output_size(input_size, filter_size, padding, stride):
        return int(1 + (input_size - filter_size + 2 * padding) / stride)

    def conv(self, inputs, filters, padding, stride):
        input_pad = np.pad(inputs, ((0, 0), (padding, padding), (padding, padding)))
        f_size = filters.shape[1]
        i_size = input_pad.shape[1]
        f = filters[np.newaxis, :, :, :] if len(filters.shape) == 3 else filters
        output_size = self.conv_output_size(inputs.shape[1], f_size, padding, stride)
        output = np.zeros((inputs.shape[0], output_size, output_size, filters.shape[-1]))
        nx = 0
        for idx in range(0, i_size - f_size)[::stride]:
            ny = 0
            for idy in range(0, i_size - f_size)[::stride]:
                windows = input_pad[:, idx:idx + f_size, idy:idy + f_size]
                output[:, nx, ny, :] = (windows[:, :, :, np.newaxis] * f).sum(axis=2).sum(axis=1)
                ny += 1
            nx += 1
        return output

    @staticmethod
    def softmax(z):
        e = np.exp(z - z.max(axis=1).reshape(-1, 1))
        return e / e.sum(axis=1).reshape(-1, 1)

    @staticmethod
    def cross_entropy_loss(a, y):
        return -np.log((a * y).sum(axis=1))

    def predict(self, x):
        return self.forward(x)[-1].argmax(axis=1)


if __name__ == '__main__':

    np.random.seed(42)

    data = load_digits()
    x = data.images
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    model = CNN(num_filters=6)
    model.fit(x_train, y_train, epochs=200)
    y_pred = model.predict(x_test)
    print(f'accuracy: {accuracy_score(y_test, y_pred):.4f}')

    wrongs_x = x_test[y_pred != y_test, :, :]
    wrongs_y = y_test[y_pred != y_test]
    wrongs_p = y_pred[y_pred != y_test]
    for i in range(len(wrongs_p)):
        plt.imshow(wrongs_x[i, :, :], cmap='gray')
        plt.title(f'true: {wrongs_y[i]}, pred: {wrongs_p[i]}')
        plt.show()
        plt.waitforbuttonpress()
        plt.close()
