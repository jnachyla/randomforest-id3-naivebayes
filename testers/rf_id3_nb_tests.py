from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from preprocessing.cars import preprocess_dataset_cars
from preprocessing.mushrooms import preprocess_dataset_mushrooms
from preprocessing.house_votes import preprocess_dataset_housevotes
from models.rf_id3_nb import RandomForest_NaivyBayes


def test_baseline_CARS():
    X,y = preprocess_dataset_cars()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

    rf = RandomForestClassifier(random_state=88, n_estimators=10)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(classification_report(y_test, y_pred))

def test_custom_CARS():
    X,y = preprocess_dataset_cars()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

    rf = RandomForest_NaivyBayes(num_trees=100, nb_at_every= 4, max_depth= 10)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(classification_report(y_test, y_pred))

def test_baseline_MUSHROOMS():
    X,y = preprocess_dataset_mushrooms()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

    rf = RandomForestClassifier(random_state=88, n_estimators=10)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(classification_report(y_test, y_pred))

def test_custom_MUSHROOMS():
    X,y = preprocess_dataset_mushrooms()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

    rf = RandomForest_NaivyBayes(n_trees=100, nb_at_every= 4, max_depth= 10)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(classification_report(y_test, y_pred))

def test_baseline_VOTES():
    X,y = preprocess_dataset_housevotes()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

    rf = RandomForestClassifier(random_state=88, n_estimators=10)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(classification_report(y_test, y_pred))

def test_custom_VOTES():
    X,y = preprocess_dataset_housevotes()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

    rf = RandomForest_NaivyBayes(num_trees=100, nb_at_every= 4, max_depth= 10)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(classification_report(y_test, y_pred))

# test_baseline_VOTES()
# test_custom_VOTES()

#test_custom_CARS()
# test_baseline_MUSHROOMS()
# test_custom_MUSHROOMS()