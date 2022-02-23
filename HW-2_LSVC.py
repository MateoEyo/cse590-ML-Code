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

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::ConvergenceWarning,ignore::RuntimeWarning')

# Load data and separate features and labels
train_set = pd.read_csv(r"C:\Users\mmate\My Drive\MS - Computer Science\CSE 590-56 Intro to Machine Learning\Homework\Code\spam_train.csv")
test_set = pd.read_csv(r"C:\Users\mmate\My Drive\MS - Computer Science\CSE 590-56 Intro to Machine Learning\Homework\Code\spam_test.csv")

X_full_train = np.array(train_set[list(train_set.columns[1:-1])])
y_full_train = np.array(train_set["class"])
X_full_test = np.array(test_set[list(test_set.columns[1:-1])])
y_full_test = np.array(test_set["class"])

# Base Result
np.random.seed(42)
ALG = LinearSVC(max_iter=10000)
ALG.fit(X_full_train, y_full_train)
print("# # #   Base result C=1   # # #")
print("Test set predictions: {}".format(ALG.predict(X_full_test)))
print("Test set accuracy: {:.2f}".format(ALG.score(X_full_test, y_full_test)))

# K-fold CV
print("\n# # #   K Fold of 5   # # #")
train_scores = []
test_scores = []
cv = KFold(n_splits=5, random_state=41, shuffle=True)
i = 1
for train_index, test_index in cv.split(train_set):
    X_train, X_test, y_train, y_test = X_full_train[train_index], X_full_train[test_index], y_full_train[train_index], y_full_train[test_index]
    np.random.seed(42)
    ALG.fit(X_train, y_train)
    train_scores.append(ALG.score(X_train, y_train))
    test_scores.append(ALG.score(X_test, y_test))
    print('Processing Fold #', i)
    print("  - Train partition shape: ", X_train.shape)
    print("  - Test partition shape: ", X_test.shape)
    print("  - Training score = ", ALG.score(X_train, y_train))
    print("  - Testing score = ", ALG.score(X_test, y_test))

    i += 1

# Print average testing score over all CV folds
print("\nAverage of scores: ", np.mean(test_scores))

# Difference between training score and testing score
print("\nDifference between training and test score: ", np.mean(train_scores)-np.mean(test_scores))

# Get param names
print("\nParam Names\n", ALG.get_params())

# Define hyperparameters and their search range
C_vals = np.logspace(0,4,10)
cross_val_score(ALG, X_full_train, y_full_train, cv=5)
hyperparameters = dict(C=C_vals)
print("\nC_vals are\n", C_vals)

clf = GridSearchCV(ALG, hyperparameters, cv=5)
clf.fit(X_train,y_train)
scores_grid = cross_val_score(ALG, X_full_train, y_full_train, cv=5)
print("\nScores:", scores_grid)
print("Average of scores: ", np.mean(scores_grid))
print("\nBest params: ",clf.best_params_)

C_vals = np.logspace(0,3,10)
print("\nC_vals are\n", C_vals)
hyperparameters = dict(C=C_vals)
clf = GridSearchCV(ALG, hyperparameters, cv=5)
clf.fit(X_train,y_train)
scores_grid = cross_val_score(ALG, X_full_train, y_full_train, cv=5)
print("\nScores:", scores_grid)
print("Average of scores: ", np.mean(scores_grid))
print("\nBest params: ",clf.best_params_)

C_vals = np.linspace(0,1,100)
print("\nC_vals are\n", C_vals)
hyperparameters = dict(C=C_vals)
clf = GridSearchCV(ALG, hyperparameters, cv=5)
clf.fit(X_train,y_train)
scores_grid = cross_val_score(ALG, X_full_train, y_full_train, cv=5)
print("\nScores:", scores_grid)
print("Average of scores: ", np.mean(scores_grid))
print("\nBest params: ",clf.best_params_)

# Fit model
np.random.seed(42)
ALG = LinearSVC(max_iter=10000, C=0.020202020202020204)
ALG.fit(X_full_test, y_full_test)
print("\nTest set predictions:\n{}".format(ALG.predict(X_full_test)))
print("\nTrain set accuracy: {:.2f}".format(ALG.score(X_full_train, y_full_train)))
print("Test set accuracy: {:.2f}".format(ALG.score(X_full_test, y_full_test)))