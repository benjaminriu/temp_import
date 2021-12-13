import numpy as np
from run_experiment import *
from get_benchmarked_methods import *

input_repository = "../preprocessed_datasets/"
output_repository = "outputs/"
benchmark_output_file = "_complete_results.csv"
prediction_output_file = None
methods = ["sklearn","mlrnet","catboost","xgboost", "lgbm", "mars"]
avoid_duplicates = False
retry_failed_exp= True

benchmark_datasets_reg = np.arange(16) #16 for regression, 16 for classification
benchmark_datasets_clf = np.arange(16) #16 for regression, 16 for classification
dataset_seeds = np.arange(1)
method_seeds = [0]
train_size = 0.8
n_features = None #all features

task_name = "regression"
regression = task_name=="regression"
run_experiment(get_benchmarked_methods(methods, regression = regression), 
                benchmark_datasets_reg, 
                task_name, 
                input_repository, 
                task_name+benchmark_output_file, 
                output_repository,
                regression = regression, 
                dataset_seeds = dataset_seeds,
                method_seeds = method_seeds,
                train_size = train_size,
                n_features = n_features,
                prediction_repository = prediction_output_file,
                avoid_duplicates = avoid_duplicates,
                retry_failed_exp= retry_failed_exp)

task_name = "classification"
regression = task_name=="regression"
run_experiment(get_benchmarked_methods(methods, regression = regression), 
                benchmark_datasets_reg, 
                task_name, 
                input_repository, 
                task_name+benchmark_output_file, 
                output_repository,
                regression = regression, 
                dataset_seeds = dataset_seeds,
                method_seeds = method_seeds,
                train_size = train_size,
                n_features = n_features,
                prediction_repository = prediction_output_file,
                avoid_duplicates = avoid_duplicates,
                retry_failed_exp= retry_failed_exp)