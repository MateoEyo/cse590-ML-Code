from re import X
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

# Create KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None,
                            n_jobs=None, n_neighbors=1, p=2, weights='uniform')

# Apply to data
clf.fit(X_train, y_train)

print("Test set predictions: {}".format(clf.predict(X_test)))
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# use in jupyter

# from re import X
# from tkinter import Y
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import mglearn

# # Get training data
# train_set = pd.read_csv("My Drive\MS - Computer Science\CSE 590-56 Intro to Machine Learning\Homework\Code\wine_train.csv")
# test_set = pd.read_csv("My Drive\MS - Computer Science\CSE 590-56 Intro to Machine Learning\Homework\Code\wine_test.csv")

# # Set train data
# X_train = np.array(train_set[list(train_set.columns[1:-1])])
# y_train = np.array(train_set["quality"])
# X_test = np.array(test_set[list(test_set.columns[1:-1])])
# y_test = np.array(test_set["quality"])

# # Create KNN algorithm
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None,
#                             n_jobs=None, n_neighbors=1, p=2, weights='uniform')

# # Apply to data
# clf.fit(X_train, y_train)

# print("Test set predictions: {}".format(clf.predict(X_test)))
# print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# # Try a whole bunch of neighbors
# print("-----------------------------`")
# training_accuracy = []
# test_accuracy = []
# neighbors_settings = range(1001, 1011)

# for n_neighbors in neighbors_settings:
#     # build the model
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors)
#     clf.fit(X_train, y_train)
#     # record training set accuracy
#     training_accuracy.append(clf.score(X_train, y_train))
#     # record generalization accuracy
#     test_accuracy.append(clf.score(X_test, y_test))
#     print("Using number: {}".format(n_neighbors))
#     print("Test set predictions: {}".format(clf.predict(X_test)))
#     print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# # fig, axes = plt.subplots(1, 3, figsize=(10,3))

# # for n_neighbors, ax in zip([1, 3, 9], axes):
# #     # the fit method returns the object self, so we can instantiate
# #     # and fit in one line
# #     clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
# #     mglearn.plots.plot_2d_separator(clf, X_train, fill=True, eps=0.5, ax=ax, alpha=.4)
# #     mglearn.discrete_scatter(X_train[:, 0], X_train[: 1], y_train, ax=ax)
# #     ax.set_title("{} neighbor(s)".format(n_neighbors))
# #     ax.set_xlabel("feature 0")
# #     ax.set_ylabel("feature 1")
# # axes[0].legend(loc=3)

