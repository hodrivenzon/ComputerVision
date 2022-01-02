import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt

print(tf.__version__)

node1 = tf.Variable(np.array([1, 2, 3, 4, 5]))
node2 = tf.Variable(np.array([1, 1, 2, 3, 5]))
print(node2.numpy())

# 5
node3 = tf.multiply(node1, node2)
print(node3.numpy())

# 6
node4 = tf.reduce_sum(node3)
print(node4.numpy())


class My_Model(object):
    def __init__(self):
        self.w = tf.Variable(tf.random.normal([1], 0.0, 1.0), dtype=tf.float32)
        self.b = tf.Variable(0, dtype=tf.float32)

    def __call__(self, x):
        return self.w * x + self.b

    def loss(self, y, target_y):
        return 0.5 * tf.reduce_sum(tf.square(y - target_y))

    def train_one_step(self, train_data, true_labels, learning_rate):
        with tf.GradientTape() as tape:
            local_loss = self.loss(self.__call__(train_data), true_labels)
        dw, db = tape.gradient(local_loss, [self.w, self.b])
        model.w.assign_sub(learning_rate * dw)
        model.b.assign_sub(learning_rate * db)

    def train(self, epochs, x_train, y_train):
        w_list = []
        b_list = []
        for i in range(epochs):
            w_list.append(model.w.numpy())
            b_list.append(model.b.numpy())
            self.train_one_step(x_train, y_train, 0.001)
        return w_list, b_list


if __name__ == "__main__":
    file_path = r'datasets/data_for_linear_regression_tf.csv'
    df = pd.read_csv(file_path, sep=",")
    data = df.values
    x_data = data[:, 0]
    label = data[:, 1]
    x_train, x_test, y_train, y_test = train_test_split(x_data, label)

    model = My_Model()
    epochs = 1000
    w_list, b_list = model.train(epochs, x_train, y_train)

    x_axis = np.array(range(epochs))
    plt.plot(x_axis, b_list, label="b")
    plt.plot(x_axis, w_list, label="w")
    plt.show()
    y_predicted = model(x_test)
    res = model.loss(y_predicted, y_test)