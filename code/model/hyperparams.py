import numpy as np
import pandas as pd

from sklearn.gaussian_process import kernels

# list of tuples of: algorithm name, if to perform gridsearch, and the hyperparam space
space = {
    'gbtree': (False, dict(
        booster = ['gbtree', 'dart'],
        gamma = np.linspace(0, 100, num=10),
        max_depth = [1, 2, 5, 6, 7, 8, 9],
        learning_rate = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2],
        n_estimators = range(1, 10),
        subsample = np.linspace(0.1, 0.9, num=5),
        objective = ['multi:softproba'], #['multi:softmax', 'multi:softproba'],
        reg_alpha = [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200],
        reg_lambda = [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200],
    )),
    'KNeighborsClassifier': (True, dict(
        # The optimal value depends on the nature of the problem
        leaf_size = np.linspace(1, 100, num = 10).astype(int),
        
        # Number of neighbors to use
        n_neighbors = np.linspace(1, 10, num = 10).astype(int),
        
        # Weight function used in prediction
        weights = ['uniform', 'distance'],
        
        # Algorithm used to compute the nearest neighbors
        algorithm= ['auto', 'ball_tree', 'kd_tree', 'brute'],
        
        # manhattan_distance (l1) or euclidean_distance (l2)
        p=[1, 2],
        
        # Parallel processing
        n_jobs = [-1],
    )),
    'SVC': (True, dict(
        C=np.linspace(0.001, 1, num=10),
        kernel=['rbf', 'poly', 'linear'],
    )),
    'RandomForestClassifier': (False, dict(
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(1, 200, num = 10)],

        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(1, 10)] + [None],

        # Minimum number of samples required to split a node
        min_samples_split = [1, 2],

        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4],

        # Method of selecting samples for training each tree
        bootstrap = [True, False],
        
        # Parallel processing
        n_jobs = [-1],
    )),
    'GaussianProcessClassifier': (False, dict(
        kernel = [kernels.RBF(), kernels.RBF() + kernels.WhiteKernel()],
        n_restarts_optimizer = [1, 2, 5],
        max_iter_predict = [10, 100, 1000],
        warm_start = [True, False],
        multi_class = ['one_vs_rest', 'one_vs_one'],
        n_jobs = [-1],
    ))
}

space['multi:softprob'] = space['gbtree']
space['XGBClassifier'] = space['gbtree']
