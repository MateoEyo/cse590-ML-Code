# Load necessary libraries
import os
import sys
import warnings
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,cross_val_predict

import pandas as pd
import numpy as np

# Load data and separate features and labels
train_set = pd.read_csv(r"C:\Users\mmate\My Drive\MS - Computer Science\CSE 590-56 Intro to Machine Learning\Homework\Code\spam_train.csv")
test_set = pd.read_csv(r"C:\Users\mmate\My Drive\MS - Computer Science\CSE 590-56 Intro to Machine Learning\Homework\Code\spam_test.csv")

X_full_train = np.array(train_set[list(train_set.columns[1:-1])])
y_full_train = np.array(train_set["class"])
X_full_test = np.array(test_set[list(test_set.columns[1:-1])])
y_full_test = np.array(test_set["class"])

# Fit model
ALG = LinearSVC(max_iter=10000, C=7719)
ALG.fit(X_full_test, y_full_test)
print("\nTest set predictions:\n{}".format(ALG.predict(X_full_test)))
print("Test set accuracy: {:.2f}".format(ALG.score(X_full_test, y_full_test)))