from re import X
from sklearn.linear_model import Lasso
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

lasso = Lasso(alpha=.01, max_iter=100000).fit(X_train, y_train)

print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ !=0)))

# Try a whole bunch of neighbors
print("-----------------------------`")
training_accuracy = []
test_accuracy = []
alpha_settings = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

for alpha in alpha_settings:
    # build the model
    lasso = Lasso(alpha=alpha, max_iter=100000)
    lasso.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(lasso.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(lasso.score(X_test, y_test))
    print("Using number: {}".format(alpha))
    print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso.coef_ !=0)))