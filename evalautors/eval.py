import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate

class ModelEvaluator(object):
    def __init__(self, model, model_name, dataset_name, eval_name):

        self.eval_name = eval_name
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model = model


    def run_binary(self,X, y, nsplits = 10):
        print("Train for dataset "+self.dataset_name)
        print("Training model... "+ self.model_name);
        scoring = {'accuracy' : make_scorer(accuracy_score),
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score),
           'f1_score' : make_scorer(f1_score)}



        kfold = StratifiedKFold(n_splits=nsplits)
        results = cross_validate(estimator=self.model,X=X,
        y=y,
        cv=kfold,
        scoring=scoring)
        print("Crossvalidation results: ",self.model_name)
        print("Accuracy: ", np.mean(results["test_accuracy"]))
        print("Precision: ", np.mean(results["test_precision"]))
        print("Precision: ", np.mean(results["test_recall"]))
        print("Recall: ", np.mean(results["test_f1_score"]))

    def run_multiclass(self,X, y, nsplits = 10):
        print("====================================")
        print(self.eval_name)
        print("Training model... "+ self.model_name);
        scoring = {
            'macro_f1_score': make_scorer(f1_score, average='macro'),
            'macro_precision': make_scorer(precision_score, average='macro'),
            'macro_recall': make_scorer(recall_score, average='macro'),
            'accuracy': make_scorer(accuracy_score)
        }

        kfold = StratifiedKFold(n_splits=nsplits)
        results = cross_validate(estimator=self.model,X=X,
        y=y,
        cv=kfold,
        scoring=scoring)
        print("Crossvalidation results: ",self.model_name)
        print("Accuracy: ", np.mean(results["test_accuracy"]))
        print("Macro Precision: ", np.mean(results["test_macro_precision"]))
        print("Macro Recall: ", np.mean(results["test_macro_recall"]))
        print("Macro F1: ", np.mean(results["test_macro_f1_score"]))