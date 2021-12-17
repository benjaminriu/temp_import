import numpy as np
from sklearn.preprocessing import StandardScaler as normalize
from sklearn.model_selection import train_test_split as tts

def dataset_loader(dataset_id, name, repository):
    return np.load(repository + name + str(dataset_id) + ".npy")
def prepare_dataset(dataset, train_size = 0.8, n_features = None, seed= False):
    kwargs = {}
    if seed or seed == 0:
        kwargs["random_state"] = seed
    X, y = dataset[:, :-1], dataset[:, -1]
    X = normalize().fit_transform(X)
    n, p = X.shape
    if type(n_features) == type(None):#all features ([:None] = [:])
        n_features = p 
    elif type(n_features) == type(1.):#percentage of all features 
        n_features = int(p * n_features)
    if n_features <= 0:#exclude last features
        n_features = p + n_features
    n_features = max(1, min(n_features, p)) #at least 1, at most p
    X = X[:,:n_features]
    X_train, X_test, y_train, y_test = tts(X, y, train_size = train_size, **kwargs)
    return X_train, X_test, y_train, y_test
def get_dataset(dataset_id, name, repository, train_size = 0.8, n_features = None, seed = False):
    return prepare_dataset(dataset_loader(dataset_id, name, repository), train_size = train_size, n_features = n_features, seed = seed)