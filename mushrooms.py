import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from rf import CustomRandomForest


from sklearn.ensemble import RandomForestClassifier

def preprocess_dataset():
    df  = pd.read_csv("./datasets/agaricus-lepiota.data", header = 0, delimiter = ',')

    df = df.sample(frac=1)

    ydf = df[['class']]
    df.drop(columns=['class'], inplace=True)
    xdf = df

    xdf = pd.get_dummies(data=xdf, columns=xdf.columns.values.tolist())

    X = xdf.values

    le = LabelEncoder()
    le.fit(ydf)
    ydf = le.transform(ydf)
    y = ydf

    return X,y


def test_baseline():
    X,y = preprocess_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

    rf = RandomForestClassifier(random_state=88, n_estimators=10)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(classification_report(y_test, y_pred))

def test_custom():
    X,y = preprocess_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

    rf = CustomRandomForest(num_trees=100)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(classification_report(y_test, y_pred))


