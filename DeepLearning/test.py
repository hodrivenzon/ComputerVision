import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class SVM():

    def __init__(self, path, ratio, seed):
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_hat = None
        self.w = None
        self.bias = 1
        self.path = path
        self.__RATIO = ratio
        self.__SEED = seed

    def import_data(self):
        ds = pd.read_csv(self.path, header=None)
        self.x = ds.iloc[1:, :-1].to_numpy()
        self.y = ds.iloc[1:, -1].to_numpy().reshape((-1, 1))
        self.y[self.y == 0] = -1
        # self.w = np.zeros((self.x.shape[1], 1))
        self.w = np.ones((self.x.shape[1], 1))

    def train_test_split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, random_state=SEED,
                                                                                train_size=RATIO)
        self.y_hat = np.zeros(shape=(self.x_test.shape[0], 1))
        return

    def hypothesis(self, x):
        return (np.dot(x, self.w) + self.bias).item(0)

    def hinge_loss(self, x, y):

        return np.max((0, 1.0 - self.hypothesis(x) * y))
        # return np.max((0, 1 - np.dot(np.dot(x, self.w) - self.bias, y)))

    def hinge_dot(self, x, y):
        # print(f"hypothesis:{self.hypothesis(x)}")
        print(f"hinge_loss:{self.hinge_loss(x, y)}")
        if self.hypothesis(x)*y < 1:
            dw = (-y * x).reshape((-1, 1))
            db = -y
        else:
            dw = np.zeros(self.w.shape)
            db = 0

        # db = y if self.hypothesis(x)*y < 1 else 0
        return dw, db

    def train_model(self):
        iters = 1
        alpha = 0.001
        for i in range(iters):
            for j in range(self.x_train.shape[0]):
                dw, db = self.hinge_dot(self.x_train[j, :], self.y_train[j, 0])
                self.w -= alpha * dw
                self.bias -= alpha * db
                if j % 100 == 0:
                    continue
                    # print(self.hinge_loss(self.x_train[j,:], self.y_train[j,0]))
            return

    def accuracy(self):
        for i in range(len(self.x_test)):
            self.y_hat[i] = self.hypothesis(self.x_test[i])
        print("SVM Accuracy:", np.mean([1 if self.y_hat[i]*self.y_test[i] > 0 else 0 for i in range(len(self.y_test))]))
        return self.y_hat

if __name__ == '__main__':
    # PATH = r'C:\Users\hodda\Downloads\diabetes.csv'
    PATH = r'C:\Users\hodda\Downloads\pima-indians-diabetes.csv'

    SEED = 1234
    RATIO = 0.75
    classifier = SVM(PATH, RATIO, SEED)
    classifier.import_data()
    classifier.train_test_split()
    classifier.train_model()
    classifier.accuracy()