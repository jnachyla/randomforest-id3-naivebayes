from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from rf import RandomForest_NaivyBayes
from cars import preprocess_dataset


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

    rf = RandomForest_NaivyBayes(num_trees=100, which_NaiveBayes = 3)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(classification_report(y_test, y_pred))


test_baseline()
test_custom()