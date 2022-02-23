# Load necessary libraries
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
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

# Base Result
train_scores = []
test_scores = []
ALG = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None,
                            n_jobs=None, n_neighbors=1, p=2, weights='uniform')
ALG.fit(X_full_train, y_full_train)                         
print("# # #   Base result C=1   # # #")
print("Test set predictions: {}".format(ALG.predict(X_full_test)))
print("Test set accuracy: {:.2f}".format(ALG.score(X_full_test, y_full_test)))

# K-fold CV
print("\n# # #   K Fold of 5   # # #")
cv = KFold(n_splits=5, random_state=41, shuffle=True)
i = 1
for train_index, test_index in cv.split(train_set):
    X_train, X_test, y_train, y_test = X_full_train[train_index], X_full_train[test_index], y_full_train[train_index], y_full_train[test_index]
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
print("\nAverage of training score: ", np.mean(train_scores))
print("Average of scores: ", np.mean(test_scores))

# Difference between training score and testing score
print("\nDifference between training and test score: ", np.mean(train_scores)-np.mean(test_scores))

# Get param names
print("\nParam Names\n", ALG.get_params())

# Define hyperparameters and their search range
K_vals = np.arange(1,2)
cross_val_score(ALG, X_full_train, y_full_train, cv=5)
hyperparameters = dict(n_neighbors=K_vals)
print("\nK_vals are\n", K_vals)

clf = GridSearchCV(ALG, hyperparameters, cv=5)
clf.fit(X_train,y_train)
scores_logreg = cross_val_score(ALG, X_full_train, y_full_train, cv=5)
print("\nBest params: ",clf.best_params_)

# Fit model
ALG = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None,
                            n_jobs=None, n_neighbors=15, p=2, weights='uniform')
ALG.fit(X_full_test, y_full_test)
print("\nTest set predictions:\n{}".format(ALG.predict(X_test)))
print("\nTrain set accuracy: {:.2f}".format(ALG.score(X_full_train, y_full_train)))
print("Test set accuracy: {:.2f}".format(ALG.score(X_full_test, y_full_test)))