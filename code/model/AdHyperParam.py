import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from . import pipeline
from . import utils

class SearchHyperParams(pipeline.Adapter):
    """ Augment the training dataset. """
    def __init__(self, algorithms_dict, show=False):
        self.show = show
        self.algorithms = algorithms_dict
        self.best = None

    def Process(self, pl):
        name = utils.nice_name(pl.model)
        if name not in self.algorithms:
            self.sayf("Skipped {}: not defined in algorithms_dict", name)
            return pl
        else:
            grid, hyp = self.algorithms[name]
        
        if grid:
            print("running grid search", end = " ... ")
            clf = GridSearchCV(pl.model, hyp, cv=pl.cv, scoring=pl.scoring,
                               n_jobs = -1)
        else:
            print("running randomized search", end = " ... ")
            clf = RandomizedSearchCV(pl.model, hyp, cv=pl.cv, scoring=pl.scoring,
                                     n_iter = 1000, n_jobs = -1)

        X = pl.Tr[pl.xCols]
        y = pl.Tr[pl.yCol]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.best = clf.fit(X, y)
        
        pl.model = self.best.best_estimator_
        self.sayf("{}-fold CV HyperParam search for {}.\nBest score: {}",
                  pl.cv, pl.model, self.best.best_score_)
        if self.show:
            self.sayf("Best parameters: {}", self.best.best_params_)
        return pl