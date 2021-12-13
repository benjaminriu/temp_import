from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import roc_auc_score as auc

def run_xgb(X_train, X_test, y_train, y_test, method_name, seed, hyper_parameters = {}, regression = True):
    try:
        if regression:
            if method_name == "XGBoost":
                from xgboost import XGBRegressor as XGB
                model = XGB(random_state = seed, objective ='reg:squarederror', verbose = False, **hyper_parameters)

            elif method_name == "CAT":
                from catboost import CatBoostRegressor as CAT
                model = CAT(random_seed=seed, logging_level='Silent', **hyper_parameters)

            elif method_name == "LGBM":
                from lightgbm.sklearn import LGBMRegressor as LGBM
                model = LGBM(random_state=seed, **hyper_parameters)
        else:
            if method_name == "XGBoost":
                from xgboost import XGBClassifier as XGB
                model = XGB(random_state = seed, verbose = False, **hyper_parameters)

            elif method_name == "CAT":
                from catboost import CatBoostClassifier as CAT
                model = CAT(random_seed=seed, logging_level='Silent', **hyper_parameters)

            elif method_name == "LGBM":
                from lightgbm.sklearn import LGBMClassifier as LGBM
                model = LGBM(random_state=seed, **hyper_parameters)

        model.fit(X_train,y_train)
        prediction = model.predict(X_test) if regression else model.predict_proba(X_test)[:,1]
        results = [r2_score(y_test, prediction)] if regression else [acc(y_test, model.predict(X_test)), auc(y_test,prediction)]
        if regression:
            results += [r2_score(y_train, model.predict(X_train))]
        else:
            results += [acc(y_train, model.predict(X_train)), auc(y_train,model.predict_proba(X_train)[:,1])]
        success = True
    except: 
        prediction = None
        results = [None, None] if regression else [None, None, None, None]
        success = False
    return success, results, prediction