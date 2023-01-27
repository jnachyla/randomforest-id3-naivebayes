import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_dataset():
    df  = pd.read_csv("./datasets/cars.data", header = 0, delimiter = ',')

    df = df.sample(frac=1)

    ydf = df[['class']]
    df.drop(columns=['class'], inplace=True)
    xdf = df

    xdf = pd.get_dummies(data=xdf, columns=['buying', 'maint', 'lug_boot', 'safety'])
    xdf['doors'].replace('5more', '5', inplace=True)
    xdf['persons'].replace('more', '5', inplace=True)
    xdf = xdf.apply(pd.to_numeric)

    X = xdf.values

    le = LabelEncoder()
    le.fit(ydf)
    ydf = le.transform(ydf)
    y = ydf

    return X,y