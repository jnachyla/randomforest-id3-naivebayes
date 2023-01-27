from sklearn.ensemble import RandomForestClassifier

from evalautors.eval import ModelEvaluator
from models.rf_id3_nb import RandomForest_NaivyBayes
from preprocessing.cars import preprocess_dataset_cars
from preprocessing.house_votes import preprocess_dataset_housevotes

def run_cars():
    X,y = preprocess_dataset_cars()

    baseline_default  = rf = RandomForestClassifier(random_state=-1)
    rf_naive_default = RandomForest_NaivyBayes()

    ModelEvaluator()