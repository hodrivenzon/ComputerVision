import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes

def hinge_loss(x, y, lamda, w, b):
    classification_term = 1 - np.dot(np.dot(x, w) - b, y)
    regularization_term = lamda * (np.linalg.norm(w) ** 2)
    return np.maximum(0, classification_term) + regularization_term


def svm_SGD(x, y, alpha, lamda):
    ww = np.ones(len(x[0]))  # initilizing weight
    bb = 1  # initilizing bias
    iterr = 10000

    # training svm
    for e in range(iterr):
        for i, val in enumerate(x):
            val1 = np.dot(x[i], ww) + bb

            if (y[i] * val1 < 1):
                ww -= alpha * (-y[i] * x[i] + 2 * lamda * ww)
                bb -= alpha * (-y[i])
            else:
                ww -= alpha * 2 * lamda * ww
                bb -= alpha * 0
        print(val1)
    #         val1 = (np.dot(x,ww) + bb)*y
    #         ww -= (alpha*np.dot(np.where(val1<1,y,0),x)+alpha*2*lamda*ww)/len(val1)
    #         bb -= alpha*np.where(val1<1,y,0).mean()
    #        ww,bb = np.where(val1<1,(ww-alpha*np.dot(y,x)+2*lamda*ww,bb-alpha*y.sum()),(ww-2*alpha*lamda*ww,bb) )

    return ww, bb


if __name__ == '__main__':

    diabetes = pd.read_csv(r'C:\Users\hodda\Downloads\diabetes.csv').values
    # diabetes = load_diabetes(as_frame=True)
    # x, y = load_diabetes(return_X_y=True, as_frame=True)

    x = diabetes[:, :-1]
    y = diabetes[:, -1]
    y = (y * 2) - 1

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    """========================================================================"""

    start_W = np.ones(len(X_train[0]))
    start_B = 1
    lamda = 0.001
    loss_val = hinge_loss(X_train, Y_train, lamda, start_W, start_B)

    W, B = svm_SGD(X_train, Y_train, 0.001, lamda)
    print('Calculated weights')
    print(W)
    print(B)

    """========================================================================"""

    predicted_y = []
    for i, val in enumerate(X_test):
        res = np.dot(X_test[i], W) + B > 1
        norm_res = res * 2 - 1
        predicted_y.append(norm_res)

    accuracy = accuracy_score(predicted_y, Y_test)
    print('test result')
    print(accuracy)

    """========================================================================"""

    # plot
    index_1 = 3
    index_2 = 2
    vals_minus1 = X_train[Y_train == -1]
    vals_plus1 = X_train[Y_train == 1]
    plt.scatter(vals_minus1[:, index_1], vals_minus1[:, index_2], s=10, c='r')
    plt.scatter(vals_plus1[:, index_1], vals_plus1[:, index_2], s=10, c='c')
    # plt.show()

#    plot hyperplane
#    x_axis = range(X_train[:,index_1].min().astype(int),X_train[:,index_1].max().astype(int))
#    y_axis = W[index_2]*x_axis
#    plt.plot(x_axis,y_axis)
