import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB

from models.decision_tree_id3 import ID3Tree


class RandomForest_NaivyBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, n_trees=50, subsample_size=None, max_depth=10000000, bootstrap=True,
                 random_state=None, nb_at_every = 2, min_samples_split = 2, split_features_fun = "None"):

        # hiperparametry drzewa ID3
        self.split_features_fun = split_features_fun
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        #hiperparametry lasu
        self.subsample_size = subsample_size
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.nb_at_every = nb_at_every
        self.trees = []



    def sample(self, X, y, random_state):
        n_rows, n_cols = X.shape

        # Sample with replacement
        if self.subsample_size is None:
            sample_size = n_rows
        else:
            sample_size = int(n_rows * self.subsample_size)

        np.random.seed(random_state)
        samples = np.random.choice(a=n_rows, size=sample_size, replace=self.bootstrap)

        return X[samples], y[samples]

    def fit(self, X, y):
        '''
        Parametry:
        ----------
        X: Zbior danych trenujacych
        y: klasa
        '''
        # Reset
        if len(self.trees) > 0:
            self.trees = []

        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values

        if isinstance(y, pd.core.series.Series):
            y = y.values

        num_built = 0

        while num_built < self.n_trees:
            if num_built % self.nb_at_every:
                gnb = MultinomialNB()
                _X, _y = self.sample(X, y, self.random_state)
                clf_nb = gnb.fit(_X, _y)
                self.trees.append(clf_nb)

            else:
                clf_id3 = ID3Tree(max_depth = self.max_depth, min_samples_split = self.min_samples_split, split_features_fun = self.split_features_fun)

                # Obtain data sample
                _X, _y = self.sample(X, y, self.random_state)
                # Train
                clf_id3.fit(_X, _y)
                # Save the classifier
                self.trees.append(clf_id3)

            num_built += 1

            if self.random_state is not None:
                self.random_state += 1

    def predict(self, X):
        # Predykcja wyznaczana dla kazdego klasyfikatora w lesie
        y = []
        for tree in self.trees:
            y.append(tree.predict(X))

        y = np.swapaxes(y, axis1=0, axis2=1)

        # Glosowanie wiekszosciowe
        predicted_classes = stats.mode(y, axis=1)[0].reshape(-1)

        return predicted_classes

