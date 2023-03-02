import warnings
import numpy as np
import pandas as pd

from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler

from . import pipeline
from . import utils


class NonCollinearFeatures(pipeline.Adapter):
    """ Set xCols based on features that do no create multicollinearity. """
    def __init__(self, keepCols = [], ignoreCols = [], vif_cutoff = 10, show=True):
        self.keepCols = keepCols
        self.ignoreCols = ignoreCols
        self.vif_cut = vif_cutoff
        self.passedCols = None
        self.failedCols = None
        self.show = show

    def find_noncollinear_features(self, df):
        """ Try to add columns one by one to dataframe without creating multicollinearity.
            Return the list of features that passed the VIF check.
            Use standard scaled values to reduce VIF.
        """

        candidate = []
        collinear = []

        passed = self.keepCols.copy()

        for col in df.columns:
            if col in passed:
                continue
            try:
                vif = utils.calc_vif(df[passed + [col]]).drop(col, axis=1)
            except:
                continue
            if vif.iloc[0, 0] < self.vif_cut:
                passed.append(col)
            else:
                candidate.append(col)
                collinear.append(vif.columns[0])

        self.failedCols = pd.DataFrame({'linearColumn': collinear}, index=candidate)
        self.passedCols = utils.calc_vif(df[passed])
        
        return self.passedCols.columns

    def Process(self, pl):
        df = pl.Tr.drop(self.ignoreCols, axis=1)
        sclr = StandardScaler().fit(df)
        sdf = pd.DataFrame(sclr.transform(df), index=df.index, columns=df.columns)
        pl.xCols = self.find_noncollinear_features(sdf)
        self.sayf("Selected {} features.", len(pl.xCols))
        if self.show:
            self.sayf("{}", list(pl.xCols))
        return pl


class AllValidFeatures(pipeline.Adapter):
    def __init__(self, ignoreCols=[], show=False):
        self.ignore = ignoreCols
        self.show = show

    def Process(self, pl):
        pl.xCols = pl.Tr.columns.drop(self.ignore)
        self.sayf("Selected {} features.", len(pl.xCols))
        if self.show:
            self.sayf("{}", list(pl.xCols))
        return pl
    

class SelectFeaturesRFE(pipeline.Adapter):
    """ Augment the training dataset. """
    def __init__(self, show=True):
        self.show = show
        self.best = None

    def Process(self, pl):
        clf = RFECV(pl.model, cv=pl.cv, n_jobs = -1, scoring = pl.scoring,
                   min_features_to_select=5)

        X = pl.Tr[pl.xCols]
        y = pl.Tr[pl.yCol]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.best = clf.fit(X, y)
        
        pl.xCols = X.columns[self.best.support_]
        self.sayf("RFE {}-fold CV with {} selected {} features.",
                  pl.cv, pl.model, len(pl.xCols))
        if self.show:
            self.sayf("{}", list(pl.xCols))
        return pl