from sklearn.ensemble import RandomForestClassifier

from evalautors.eval import ModelEvaluator
from models.decision_tree_id3 import ID3Tree
from models.rf_id3_nb import RandomForest_NaivyBayes

from preprocessing.house_votes import preprocess_dataset_housevotes
from sklearn.naive_bayes import MultinomialNB



def run_default_votes():
    X,y = preprocess_dataset_housevotes()

    baseline_default = RandomForestClassifier(n_estimators=100,
        criterion="entropy",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    )

    rf_naive_default = RandomForest_NaivyBayes(n_trees=100,
                                               min_samples_split=2,
                                               random_state=None)

    evaluator_baseline = ModelEvaluator(model_name="RF Baseline Default", model=baseline_default, eval_name="Default Parameters", dataset_name="votes")
    evaluator_baseline.run_multiclass(X=X,y=y)

    evaluator_custom = ModelEvaluator(model_name="RF Custom Default", model=rf_naive_default,
                                         eval_name="Default Parameters", dataset_name="votes")
    evaluator_custom.run_binary(X=X, y=y)

def run_default_on_simple_votes():
    X,y = preprocess_dataset_housevotes()

    tree = ID3Tree()

    nb = MultinomialNB()

    evaluator_tree = ModelEvaluator(model_name="Tree ID3", model=tree, eval_name="Tree ID3 Default Parameters", dataset_name="votes")
    evaluator_tree.run_binary(X=X,y=y)

    evaluator_custom = ModelEvaluator(model_name="Naive Bayess", model=nb,
                                         eval_name="NB Default Parameters", dataset_name="votes")
    evaluator_custom.run_binary(X=X, y=y)

def run_hiperparameters_votes():
    model_name = "Naive Random Forest "
    X,y = preprocess_dataset_housevotes()

    for n_trees in [10,20,50,100,150,200]:
        rf_naive_default = RandomForest_NaivyBayes(n_trees=n_trees,
                                                   min_samples_split=2,
                                                   random_state=None)

        evaluator_custom = ModelEvaluator(model_name=model_name+str(n_trees), model=rf_naive_default,
                                          eval_name="RF different Num Trees experiment", dataset_name="votes")
        evaluator_custom.run_binary(X=X, y=y)

    for max_depth in [5,10,20,30,50,100,150, 200]:
        rf_naive_default = RandomForest_NaivyBayes(n_trees=100,
                                                   min_samples_split=2,
                                                   max_depth=max_depth,
                                                   random_state=None)

        evaluator_custom = ModelEvaluator(model_name=model_name+str(max_depth), model=rf_naive_default,
                                          eval_name="RF different max depth experiment", dataset_name="votes")
        evaluator_custom.run_binary(X=X, y=y)

    for fun in ("None",'log2','sqrt' ):
        rf_naive_default = RandomForest_NaivyBayes(n_trees=100,
                                                   min_samples_split=2,
                                                   split_features_fun = fun,
                                                   random_state=None)

        evaluator_custom = ModelEvaluator(model_name="Custom RF with ntrees = "+str(fun), model=rf_naive_default,
                                          eval_name="RF different fun experiment", dataset_name="votes")
        evaluator_custom.run_binary(X=X, y=y)


    for nb_at_every in [2,4,6,10]:
        rf_naive_default = RandomForest_NaivyBayes(n_trees=100,
                                                   min_samples_split=2,
                                                   nb_at_every=nb_at_every,
                                                   random_state=None)

        evaluator_custom = ModelEvaluator(model_name="Custom RF with ntrees = "+str(nb_at_every), model=rf_naive_default,
                                          eval_name="RF different NB at every experiment", dataset_name="votes")
        evaluator_custom.run_binary(X=X, y=y)

    for min_samples_split in [2,4,10]:
        rf_naive_default = RandomForest_NaivyBayes(n_trees=100,
                                                   min_samples_split=min_samples_split,
                                                   random_state=None)

        evaluator_custom = ModelEvaluator(model_name=model_name+str(min_samples_split), model=rf_naive_default,
                                          eval_name="RF different min samples split experiment", dataset_name="votes")
        evaluator_custom.run_binary(X=X, y=y)



run_default_votes()
print("***************************")
run_default_on_simple_votes()
print("****************************")
run_hiperparameters_votes()