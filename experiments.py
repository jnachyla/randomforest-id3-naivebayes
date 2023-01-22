import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate


class ExperimentRunner(object):
    def __init__(self, model, model_name, X, y):

        self.model_name = model_name
        self.model = model

        self.y = y
        self.X = X


    def run(self, nsplits = 10):
        print("Training model... "+ self.model_name);
        scoring = {'accuracy' : make_scorer(accuracy_score),
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score),
           'f1_score' : make_scorer(f1_score)}

        kfold = KFold(n_splits=nsplits)
        results = cross_validate(estimator=rf,X=X,
        y=y,
        cv=kfold,
        scoring=scoring)
        print("Crossvalidation results: ",self.model_name)
        print("Accuracy: ", np.mean(results["test_accuracy"]))
        print("Precision: ", np.mean(results["test_precision"]))
        print("Precision: ", np.mean(results["test_recall"]))
        print("Recall: ", np.mean(results["test_f1_score"]))