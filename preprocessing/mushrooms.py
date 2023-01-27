import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_dataset_mushrooms():
    df  = pd.read_csv("../preprocessing/agaricus-lepiota.data", header = 0, delimiter = ',')

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



