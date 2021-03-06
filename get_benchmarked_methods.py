import importlib
def get_benchmarked_methods(methods, regression = True):
    #methods format: Category Name (str), Method Name (str), Method Function (str), Hyparameters name (str), Hyperparameters (dict)
    
    torch_available = importlib.util.find_spec('torch') is not None #conda install -c pytorch pytorch
    xgboost_available = importlib.util.find_spec('xgboost') is not None #conda install -c conda-forge xgboost
    catboost_available = importlib.util.find_spec('catboost') is not None #conda install -c conda-forge catboost
    lgbm_available = importlib.util.find_spec('lightgbm') is not None #conda install -c conda-forge lightgbm
    mars_available = importlib.util.find_spec('pyearth') is not None #conda install -c conda-forge sklearn-contrib-py-earth
    optuna_available = importlib.util.find_spec('optuna') is not None #conda install -c conda-forge optuna
    #else excluded from the benchmark
    
    test_sklearn = "sklearn" in methods
    test_rf = "RF" in methods or test_sklearn
    test_mlrnet = "mlrnet" in methods and torch_available
    test_regularnet = "regularnet" in methods and torch_available
    test_xgboost = "xgboost" in methods and xgboost_available
    test_catboost = "catboost" in methods and catboost_available
    test_lgbm = "lgbm" in methods and lgbm_available
    test_mars = "mars" in methods and mars_available
    test_hpo = "HPO" in methods and optuna_available
    test_nohpo = "HPO_only" not in methods
    test_fast_mlrnet = "fast_mlrnet" in methods or test_mlrnet
    test_fast_regularnet = "fast_regularnet" in methods or test_regularnet
    
    baseline_name = "Baseline"
    lm_name = "GLM"
    QDA_name = "QDA"
    tree_name = "TREE"
    ensemble_name = "RF"
    spline_name = "MARS"
    svm_name = "SVM"
    nn_name = "NN"
    xgb_name = "GBDT"
    mlrnet_name = "mlrnet"

    xgb_experiment = "run_xgb"
    sklearn_experiment = "run_sklearn"
    mlrnet_experiment = "run_mlrnet"
    regularnet_experiment = "run_regularnet"
    hpo_experiment = "run_HPO"

    regressor_methods = []
    classifier_methods = []

    regressor_methods += [(baseline_name, "Intercept", sklearn_experiment, "nohp", {}),
                         (lm_name, "Ridge", sklearn_experiment, "nohp", {}),
                         (lm_name, "Lasso", sklearn_experiment, "nohp", {}),
                         (lm_name, "Enet", sklearn_experiment, "nohp", {}),
                         (tree_name, "CART", sklearn_experiment, "nohp", {}),
                         (ensemble_name, "XRF", sklearn_experiment, "nohp", {}),
                         (xgb_name, "xgb_sklearn", sklearn_experiment, "nohp", {}),
                         (svm_name, "Kernel", sklearn_experiment, "nohp", {}),
                         (svm_name, "NuSVM", sklearn_experiment, "nohp", {}),
                         (nn_name, "MLP_sklearn", sklearn_experiment, "nohp", {})] * test_sklearn * test_nohpo
    
    
    regressor_methods += [(ensemble_name, "RF", sklearn_experiment, "nohp", {})] * test_rf * test_nohpo
    regressor_methods += [(spline_name, "MARS", sklearn_experiment, "nohp", {})] * test_mars * test_nohpo
    regressor_methods += [(xgb_name, "XGBoost", xgb_experiment, "nohp", {})] * test_xgboost * test_nohpo
    regressor_methods += [(xgb_name, "CAT", xgb_experiment, "nohp", {})] * test_catboost * test_nohpo
    regressor_methods += [(xgb_name, "LGBM", xgb_experiment, "nohp", {})] * test_lgbm * test_nohpo
    
    regressor_methods += [(mlrnet_name, "mlrnetfast", mlrnet_experiment, "nohp", {})] * test_fast_mlrnet * test_nohpo
    regressor_methods += [(mlrnet_name, "mlrnetstandard", mlrnet_experiment, "nohp", {})] * test_mlrnet * test_nohpo
    regressor_methods += [(mlrnet_name, "mlrnetresblock", mlrnet_experiment, "nohp", {})] * test_mlrnet * test_nohpo
    
    regressor_methods += [(nn_name, "regularnetfast", regularnet_experiment, "nohp", {})] * test_fast_regularnet * test_nohpo
    regressor_methods += [(nn_name, "regularnetstandard", regularnet_experiment, "nohp", {})] * test_regularnet * test_nohpo
    regressor_methods += [(nn_name, "regularnetresblock", regularnet_experiment, "nohp", {})] * test_regularnet * test_nohpo

    regressor_methods += [(xgb_name, "CAT", hpo_experiment, "hpo", {"function":xgb_experiment})]* test_hpo * test_catboost
    regressor_methods += [(xgb_name, "XGBoost", hpo_experiment, "hpo", {"function":xgb_experiment})]* test_hpo * test_xgboost
    regressor_methods += [(ensemble_name, "RF", hpo_experiment, "hpo", {"function":sklearn_experiment})]* test_hpo * test_rf
    regressor_methods += [(mlrnet_name, "mlrnetHPO", hpo_experiment, "hpo", {"function":mlrnet_experiment})] * test_hpo * test_mlrnet
    
    classifier_methods += [(baseline_name, "Intercept", sklearn_experiment, "nohp", {}),
                         (lm_name, "Ridge", sklearn_experiment, "nohp", {}),
                         (lm_name, "LinearRidge", sklearn_experiment, "nohp", {}),
                         (lm_name, "Lasso", sklearn_experiment, "nohp", {}),
                         (lm_name, "Enet", sklearn_experiment, "nohp", {}),
                         (QDA_name, "QDA", sklearn_experiment, "nohp", {}),
                         (tree_name, "CART", sklearn_experiment, "nohp", {}),
                         (tree_name, "XCART", sklearn_experiment, "nohp", {}),
                         (ensemble_name, "XRF", sklearn_experiment, "nohp", {}),
                         (xgb_name, "xgb_sklearn", sklearn_experiment, "nohp", {}),
                         (xgb_name, "ADABoost", sklearn_experiment, "nohp", {}),
                         (nn_name, "MLP_sklearn", sklearn_experiment, "nohp", {})] * test_sklearn * test_nohpo
    
    classifier_methods += [(ensemble_name, "RF", sklearn_experiment, "nohp", {})] * test_rf * test_nohpo
    classifier_methods += [(xgb_name, "XGBoost", xgb_experiment, "nohp", {})] * test_xgboost * test_nohpo
    classifier_methods += [(xgb_name, "CAT", xgb_experiment, "nohp", {})]  * test_catboost * test_nohpo
    classifier_methods += [(xgb_name, "LGBM", xgb_experiment, "nohp", {})] * test_lgbm * test_nohpo
    
    classifier_methods += [(mlrnet_name, "mlrnetfast", mlrnet_experiment, "nohp", {})] * test_fast_mlrnet * test_nohpo
    classifier_methods += [(mlrnet_name, "mlrnetstandard", mlrnet_experiment, "nohp", {})] * test_mlrnet * test_nohpo
    classifier_methods += [(mlrnet_name, "mlrnetresblock", mlrnet_experiment, "nohp", {})] * test_mlrnet * test_nohpo
    
    classifier_methods += [(nn_name, "regularnetfast", regularnet_experiment, "nohp", {})] * test_fast_regularnet * test_nohpo
    classifier_methods += [(nn_name, "regularnetstandard", regularnet_experiment, "nohp", {})] * test_regularnet * test_nohpo
    classifier_methods += [(nn_name, "regularnetresblock", regularnet_experiment, "nohp", {})] * test_regularnet * test_nohpo
    
    classifier_methods += [(xgb_name, "CAT", hpo_experiment, "hpo", {"function":xgb_experiment})]* test_hpo * test_catboost
    classifier_methods += [(xgb_name, "XGBoost", hpo_experiment, "hpo", {"function":xgb_experiment})]* test_hpo * test_xgboost
    classifier_methods += [(ensemble_name, "RF", hpo_experiment, "hpo", {"function":sklearn_experiment})]* test_hpo * test_rf
    classifier_methods += [(mlrnet_name, "mlrnetHPO", hpo_experiment, "hpo", {"function":mlrnet_experiment})] * test_hpo * test_mlrnet
    
    if regression: return regressor_methods
    else: return classifier_methods 
def get_run_(function):
    if function == "run_sklearn": from run_sklearn import run_sklearn
    if function == "run_xgb": from run_xgb import run_xgb
    if function == "run_regularnet": from run_regularnet import run_regularnet
    if function == "run_mlrnet": from run_mlrnet import run_mlrnet
    if function == "run_HPO": from run_HPO import run_HPO
    return eval(function)