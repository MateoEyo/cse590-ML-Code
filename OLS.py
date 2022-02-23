from re import X
from sklearn.linear_model import LinearRegression
from tkinter import Y
import numpy as np
import pandas as pd

# Get training data
train_set = pd.read_csv("wine_train.csv")
test_set = pd.read_csv("wine_test.csv")

# Set train data
X_train = np.array(train_set[list(train_set.columns[1:-1])])
y_train = np.array(train_set["quality"])
X_test = np.array(test_set[list(test_set.columns[1:-1])])
y_test = np.array(test_set["quality"])

lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))