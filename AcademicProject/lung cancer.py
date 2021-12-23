
# DONE


# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


def svm_trainer(kernel, xt, yt, xe, ye):
    # Training the SVM model on the Training set
    classifier = SVC(kernel=kernel, random_state=0)
    classifier.fit(xt, yt)

    # Predicting a new result
    classifier.predict(sc.transform([[0, 52, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2]]))

    # Predicting the Test set results
    y_pred = classifier.predict(xe)
    np.concatenate((y_pred.reshape(len(y_pred), 1), ye.reshape(len(ye), 1)), 1)

    # Making the Confusion Matrix
    cm = confusion_matrix(ye, y_pred)
    score = accuracy_score(ye, y_pred)
    return {kernel: score}


# Importing the dataset
# Gender 11 -> male
#         0 -> Female
dataset = pd.read_csv('lung cancer.csv')
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# rbf -> 0.8974358974358975
# poly -> 0.9102564102564102
# sigmoid -> 0.9358974358974359    best kernel for this model
# linear -> 0.9358974358974359     best kernel for this model
kernels = ['rbf', 'poly', 'linear', 'sigmoid']
accuracies = list()
for i in kernels:
    accuracies.append(svm_trainer(i, X_train, y_train, X_test, y_test))

print(accuracies)
# # Visualising the Training set results
# from matplotlib.colors import ListedColormap
#
# X_set, y_set = sc.inverse_transform(X_train), y_train
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
#                      np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
# plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
# plt.title('SVM (Training set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()

# Visualising the Test set results
# from matplotlib.colors import ListedColormap
#
# X_set, y_set = sc.inverse_transform(X_test), y_test
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
#                      np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
# plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
# plt.title('SVM (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
