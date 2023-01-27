import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_dataset_housevotes():
    df  = pd.read_csv("./datasets/house-votes-84.data", header = None, delimiter = ',')

    df = df.sample(frac=1)

    # zastepujemy brakujace ? wartosciami NAN
    df.replace("?", np.nan, inplace=True)

    # zamieniamy 0 i 1 na wartości liczbowe, inaczej nie dziala mediana w pandas
    df.replace("y", 1, inplace=True)
    df.replace("n", 0, inplace=True)

    # usuwamy wiersze wartości z wiecej niż 5 pustymi kolumnami
    df = df[df.isnull().sum(axis=1) < 5]

    # zastepujemy brakujace wartości medianą
    df = df.fillna(df.median())

    ydf = df.iloc[:,0]
    df.drop(columns=[0], inplace=True)

    xdf = df
    xdf = xdf.apply(pd.to_numeric)

    X = xdf.values

    le = LabelEncoder()
    le.fit(ydf)
    ydf = le.transform(ydf)
    y = ydf

    return X,y
