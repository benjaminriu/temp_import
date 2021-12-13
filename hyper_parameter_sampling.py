#hyper_parameter_sampling
from copy import deepcopy
def sample_RF(trial):
    sample = {"n_estimators":100}
    sample.update({"max_features" : trial.suggest_categorical("max_features", ['auto', 'sqrt', "log2"]),
    "max_depth" : trial.suggest_int("max_depth", 2, 100, log = True),
    "max_leaf_nodes" : trial.suggest_int("max_leaf_nodes", 2, 1024, log = True),
    "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 1, 16, log = True),
    "bootstrap" : trial.suggest_categorical("bootstrap", [True, False]),
    "max_samples" : trial.suggest_float("max_samples", 0.05, 1.)})
    return sample

def translate_RF(trial_params):
    params = {"n_estimators":100}
    params.update(deepcopy(trial_params))
    return params

def sample_XGBoost(trial):
    sample = {"learning_rate" : trial.suggest_float("learning_rate", 1e-7 ,1, log = True ),
    "max_depth" : trial.suggest_int("max_depth", 1, 10),
    "subsample" : trial.suggest_float("subsample", 0.2, 1),
    "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.2, 1),
    "colsample_bylevel" : trial.suggest_float("colsample_bylevel", 0.2, 1),
    "min_child_weight" : trial.suggest_float("min_child_weight", 1e-16,1e5, log = True),
    "reg_alpha" : trial.suggest_float("reg_alpha", 1e-16, 1e2, log = True),
    "reg_lambda" : trial.suggest_float("reg_lambda", 1e-16, 1e2, log = True),
    "gamma" : trial.suggest_float("gamma", 1e-16, 1e2, log = True)}
    return sample

def translate_XGBoost(trial_params):
    return deepcopy(trial_params)

def sample_CAT(trial):
    sample = {"learning_rate" : trial.suggest_float("learning_rate", 1e-5, 1, log = True),
    "random_strength" : trial.suggest_int("random_strength", 1, 20),
    "bagging_temperature" : trial.suggest_float("bagging_temperature", 0, 1),
    "l2_leaf_reg" : trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
    "leaf_estimation_iterations" : trial.suggest_int("leaf_estimation_iterations", 1, 20)}
    return sample
    
def translate_CAT(trial):
    return deepcopy(trial_params)

def sample_mlrnetHPO(trial):
    parameters = {}
    parameters["depth"] = trial.suggest_int("depth", 1, 5, log=False)
    max_width = 1024 if parameters["depth"] > 2 else 4096 
    parameters["width"]= trial.suggest_int("width", 16, max_width, log=True)
    parameters["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    parameters["max_iter"] = trial.suggest_int("max_iter", 20, 2000, log=True)
    parameters["n_permut"] = trial.suggest_categorical("n_permut", [0, 1, 64])
    parameters["closeform_intercept"] = trial.suggest_categorical("closeform_intercept", [True, False])
    return parameters

def translate_mlrnetHPO(trial_params):
    import architectures
    return {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":trial_params["learning_rate"], "total_steps" : trial_params["max_iter"]},
                "max_iter":trial_params["max_iter"],
                "learning_rate":trial_params["learning_rate"]/10., #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":trial_params["width"],"depth":trial_params["depth"]},
                "closeform_intercept" : trial_params["closeform_intercept"]
                }