import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def gradient_boosting_regressor(X_train, Y_train, epochs):
    models = []
    gradient = Y_train
    for _ in range(epochs):
        h = DecisionTreeRegressor(max_depth=1).fit(X_train, gradient)
        gradient -= h.predict(X_train)
        models.append(h)

    def gradient_boosting_predict(X):
        # Y_pred = np.zeros_like(X.shape[0])
        Y_pred = np.zeros(X.shape[0])
        for model in models:
            Y_pred += model.predict(X)
        return Y_pred

    return gradient_boosting_predict


def preprocess_fish(df):
    df = pd.get_dummies(df)
    Y = df.values[:, 0]
    X = df.values[:, 1:]
    X = RobustScaler().fit_transform(X)
    return X, Y


if __name__ == "__main__":

    df = pd.read_csv(r'datasets\csv\Fish.csv')
    X, Y = preprocess_fish(df)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
    gradient_boosting_predict = gradient_boosting_regressor(X_train, Y_train, 100)
    print(mean_absolute_error(Y_train, gradient_boosting_predict(X_train)))
    print(mean_absolute_error(Y_test, gradient_boosting_predict(X_test)))

