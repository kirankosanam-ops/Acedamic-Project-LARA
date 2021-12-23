import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.svm import SVC


def svm_trainer(kernel, X_train, y_train, X_test, y_test):
    # Training the SVM model on the Training set
    classifier = SVC(kernel=kernel, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred_svm = classifier.predict(X_test)

    # confusion matrix
    cm_test = confusion_matrix(y_pred_svm, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    score = (cm_train[0][0] + cm_train[1][1]) / len(y_train)
    return {kernel: score}


df = pd.read_csv('cleveland.csv', header=None)

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']

df.isnull().sum()

df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['sex'] = df.sex.map({0: 'female', 1: 'male'})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())
df['sex'] = df.sex.map({'female': 0, 'male': 1})

# data preprocessing
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# scaling/transforming data
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# sigmoid -> 0.8512396694214877
# linear -> 0.8553719008264463
# rbf -> 0.9256198347107438
# poly -> 0.9462809917355371


kernels = ['rbf', 'poly', 'linear', 'sigmoid']
accuracies = list()
for i in kernels:
    accuracies.append(svm_trainer(i, X_train, y_train, X_test, y_test))
print(accuracies)
