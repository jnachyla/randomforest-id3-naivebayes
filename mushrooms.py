import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from rf_ID3_NB import RandomForest_NaivyBayes
from sklearn.ensemble import RandomForestClassifier

def preprocess_dataset_mushrooms():
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



