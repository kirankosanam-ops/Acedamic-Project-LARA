from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss

_optimizers = ['adam', 'AdaGrad', 'SGD', 'Adadelta', 'RMSprop']


def build_model(optimizer):
    mm = Sequential()
    mm.add(Dense(13, input_dim=13, activation='relu'))
    mm.add(Dense(8, activation='relu'))
    mm.add(Dense(8, activation='relu'))
    mm.add(Dense(1, activation='sigmoid'))
    mm.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # adam, AdaGrad, SGD, Adadelta, RMSprop
    return mm


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
accuracies = dict()
# building a classification model with different optimizers
for i in _optimizers:
    model = build_model(optimizer=i)
    model.fit(X, y, epochs=150, batch_size=10)
    _, accuracy = model.evaluate(X_test, y_test)
    accuracies[i] = accuracy
    print(f'Optimizer = {i} accuracy = {accuracy}')
print(accuracies)
print([{optimizer:accuracy} for optimizer, accuracy in accuracies.items() if accuracy == max(accuracies.values())][0])


# epochs - 150/ batch size - 10 -> 0.8196721076965332
