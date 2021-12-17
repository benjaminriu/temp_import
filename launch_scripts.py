import numpy as np
from reorder_features import reorder_features
order_output_file = "../feature_reorder_index/"
input_repository = "../preprocessed_datasets/"
output_repository = "../reordered_datasets/"
datasets = np.arange(16)
rewrite = True
max_samples = 1000
reorder_features(datasets,
                    order_output_file, 
                    input_repository, 
                    output_repository, 
                    regression = True,
                    max_samples = max_samples,
                    rewrite = rewrite)
reorder_features(datasets,
                    order_output_file, 
                    input_repository, 
                    output_repository, 
                    regression = False,
                    max_samples = max_samples,
                    rewrite = rewrite)