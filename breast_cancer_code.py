import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

data = pd.read_csv("breast_cancer.csv")
# data.head()
# data.shape

data.isnull().values.any()
data.isnull().sum()

X = data.iloc[:, 1:10].values
# features[:, 3]

y = data.iloc[:, 10].values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 5:6])
X[:, 5:6] = imputer.transform(X[:, 5:6])
# features[:, 5]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,
                                                    test_size=0.20)

# Feature scaling the features for easy calculation
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# sigmoid -> 0.9214285714285714
# poly -> 0.9285714285714286
# linear -> 0.9714285714285714
# rbf -> 0.9714285714285714

clf_sigmoid = SVC(kernel="poly", random_state=0)
clf_sigmoid.fit(X_train, y_train)
label_pred_sigmoid = clf_sigmoid.predict(X_test)

clf_rbf = SVC(kernel="linear", random_state=0)
clf_rbf.fit(X_train, y_train)
label_pred_rbf = clf_rbf.predict(X_test)


cm = confusion_matrix(label_pred_sigmoid, label_pred_rbf)
score = clf_sigmoid.score(X_test, y_test)
print('poly', score)

score = clf_rbf.score(X_test, y_test)
print('linear', score)

# custom input
vals1 = [6, 2, 5, 3, 9, 4, 7, 2, 2]  # 6, 2, 5, 3, 9, 4, 7, 2, 2,[4]
vals = [4, 1, 1, 3, 2, 1, 3, 1, 1]  # 4,1,1,3,2,1,3,1,1,[2]
a = sc.transform(np.array(vals).reshape(1, -1))
# print(a)
# 2 for Benign and 4 for Malignant
cancer_type = {2: 'Benign', 4: 'Malignant'}
cancer_pred = clf_sigmoid.predict(a)
print(cancer_type[cancer_pred[0]])
