import numpy as np
import pandas as pd

from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from decisionTree_ID3 import ID3Tree
from sklearn.naive_bayes import GaussianNB


class RandomForest_NaivyBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, num_trees=5, subsample_size=None, max_depth=5, max_features=None, bootstrap=True,
                 random_state=None, which_NaiveBayes = 2, min_samples_split = 2):
        self.n_trees = num_trees
        self.subsample_size = subsample_size
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.which_NaiveBayes = which_NaiveBayes
        self.min_samples_split = min_samples_split
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
        which_NaiveBayes = 1

        while num_built < self.n_trees:
            if which_NaiveBayes == self.which_NaiveBayes:
                gnb = GaussianNB()
                _X, _y = self.sample(X, y, self.random_state)
                clf_nb = gnb.fit(_X, _y)
                self.trees.append(clf_nb)
                which_NaiveBayes = 1

            else:
                split_features_funInTree = self.randomSplitFeatures()
                clf_id3 = ID3Tree(max_depth = self.max_depth, min_samples_split = self.min_samples_split, split_features_fun = split_features_funInTree)

                # Obtain data sample
                _X, _y = self.sample(X, y, self.random_state)
                # Train
                clf_id3.fit(_X, _y)
                # Save the classifier
                self.trees.append(clf_id3)
                which_NaiveBayes += 1

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

    def randomSplitFeatures(self):
        featuresSplit_list = ["None", "log2", "sqrt"]
        return np.random.choice(featuresSplit_list)
