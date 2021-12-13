from get_benchmarked_methods import get_run_
from copy import deepcopy
from hyper_parameter_sampling import *
from functools import partial
from sklearn.model_selection import KFold
import optuna
import numpy as np
def run_HPO(X_train, X_test, y_train, y_test, method_name, seed, hyper_parameters = {}, regression = True,
           n_trials = 100, timeout = 6000, kfold_seeds= [0], shuffle=True, cvs = 5):
    HPO_parameters = deepcopy(hyper_parameters)
    function = HPO_parameters.pop("function")
    
    #pass other hyper_parameters as parameters to run_HPO
    if HPO_parameters != {}: return run_HPO(X_train, 
                       X_test, 
                       y_train, 
                       y_test, 
                       method_name, 
                       seed, 
                       hyper_parameters = {"function":function}, 
                       regression = regression, **HPO_parameters)
    
    run_function = get_run_(function)
    sample_function = eval('sample_'+method_name)
    translate_function = eval('translate_'+method_name)
    objective = partial(objective_func, 
                   run_function = run_function, 
                   sample_function = sample_function, 
                   translate_function = translate_function, 
                   method_name = method_name, 
                   X_train = X_train, 
                   y_train = y_train, 
                   method_seed = seed,
                   kfold_seeds= kfold_seeds,
                   shuffle=shuffle,
                   cvs = cvs, 
                   regression = regression)
        
    optuna_params = run_optuna(objective, n_trials = n_trials, timeout = timeout)
    best_hyper_parameters = translate_function(optuna_params[0])
    
    #refit
    
    success, results, prediction = run_function(X_train, X_test, y_train, y_test, method_name, seed, hyper_parameters = best_hyper_parameters, regression = regression)
    return success, results, prediction
    
def run_optuna(objective, n_trials = 100, timeout = 6000):
    study = optuna.create_study( directions=["maximize"])
    study.optimize(objective, n_trials=n_trials, timeout = timeout)
    params = [trial.params for trial in study.trials]
    return [params[i] for i in np.argsort([trial.value for trial in study.trials])][::-1]#params from best to worst
    
def objective_func(trial, 
                   run_function, 
                   sample_function, 
                   translate_function, 
                   method_name, 
                   X_train, 
                   y_train, 
                   method_seed,
                   regression = True,
                   kfold_seeds= [0],
                   shuffle=True,
                   cvs = 10):

    trial_params = sample_function(trial)
    hyper_parameters = translate_function(trial_params)
    performance = []
    for kfold_seed in kfold_seeds:
        kf = KFold(n_splits=cvs,random_state=kfold_seed, shuffle=shuffle)
        for train_index, valid_index in kf.split(X_train):
            success, results, prediction = run_function(X_train = X_train[train_index], 
                                                        X_test = X_train[valid_index], 
                                                        y_train = y_train[train_index], 
                                                        y_test = y_train[valid_index], 
                                                        method_name = method_name, 
                                                        seed = method_seed,
                                                        hyper_parameters = deepcopy(hyper_parameters))
            if not success:
                return -1.
            else: performance.append(results[0])#R2-score or ACCURACY
    return np.mean(performance)