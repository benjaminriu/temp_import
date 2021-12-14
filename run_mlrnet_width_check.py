import numpy as np
from run_experiment import *

input_repository = "../preprocessed_datasets/"
output_repository = "outputs/"
benchmark_output_file = "_cholesky_test.csv"
prediction_output_file = None
methods = ["RF","mlrnet"]
avoid_duplicates = True
retry_failed_exp= True

benchmark_datasets_reg = [14] #16 for regression, 16 for classification
benchmark_datasets_clf = [13] #16 for regression, 16 for classification
dataset_seeds = np.arange(1)
method_seeds = [0]
train_size = 0.8
n_features = None #all features

task_name = "regression"
regression = task_name=="regression"

mlrnet_name = "mlrnet"
mlrnet_experiment = "run_mlrnet"

methods = [(mlrnet_name, "mlrnetw2048", mlrnet_experiment, "nohp", {}),
          (mlrnet_name, "mlrnetw4096", mlrnet_experiment, "nohp", {}),
          (mlrnet_name, "mlrnetw8192", mlrnet_experiment, "nohp", {}),
          (mlrnet_name, "mlrnetw16384", mlrnet_experiment, "nohp", {}),
          (mlrnet_name, "mlrnetw32768", mlrnet_experiment, "nohp", {})]
run_experiment(methods,
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