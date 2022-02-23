from re import X
from sklearn.linear_model import Ridge
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

ridge = Ridge(alpha=.1).fit(X_train, y_train)

print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

# Try a whole bunch of neighbors
print("-----------------------------`")
training_accuracy = []
test_accuracy = []
alpha_settings = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for alpha in alpha_settings:
    # build the model
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(ridge.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(ridge.score(X_test, y_test))
    print("Using number: {}".format(alpha))
    print("Test set predictions: {}".format(ridge.predict(X_test)))
    print("Test set accuracy: {:.2f}".format(ridge.score(X_test, y_test)))