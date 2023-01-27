import pandas as pd
from models.rf_id3_nb import RandomForest_NaivyBayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def preprocess_dataset_weather():
    df  = pd.read_csv("../datasets/weather.csv", header = 0, delimiter = ',')

    ydf = df[['play']]
    df.drop(columns=['play'], inplace=True)
    xdf = df

    ord_enc = OrdinalEncoder()
    xdf["outlook"] = ord_enc.fit_transform(xdf[["outlook"]])
    ord_enc = OrdinalEncoder()
    xdf["temperature"] = ord_enc.fit_transform(xdf[["temperature"]])
    ord_enc = OrdinalEncoder()
    xdf["humidity"] = ord_enc.fit_transform(xdf[["humidity"]])
    ord_enc = OrdinalEncoder()
    xdf["wind"] = ord_enc.fit_transform(xdf[["wind"]])

    #xdf = pd.get_dummies(data=xdf, columns=xdf.columns.values.tolist())

    X = xdf.values

    le = LabelEncoder()
    le.fit(ydf)
    ydf = le.transform(ydf)
    y = ydf

    return X,y


def test_baseline_WEATHER():
    X,y = preprocess_dataset_weather()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

    rf = RandomForestClassifier(random_state=88, n_estimators=10)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(classification_report(y_test, y_pred))

def test_custom_WEATHER():
    X,y = preprocess_dataset_weather()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

    rf = RandomForest_NaivyBayes(num_trees=100, nb_at_every= 4, max_depth= 10)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(classification_report(y_test, y_pred))

