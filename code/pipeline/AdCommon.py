import numpy as np
import pandas as pd

from . import pipeline
from . import utils


class SetYCol(pipeline.Adapter):
    def __init__(self, ycol):
        self.ycol = ycol
    def Process(self, pl):
        pl.yCol = self.ycol
        print("'%s'" %pl.yCol, end = " ")
        return pl

class DropCol(pipeline.Adapter):
    def __init__(self, col):
        self.col = col
    def Process(self, pl):
        pl.Tr = pl.Tr.drop(self.col, axis=1)
        pl.Ts = pl.Ts.drop(self.col, axis=1)
        if pl.xCols is not None and self.col in pl.xCols:
            pl.xCols = pl.xCols.drop(self.col)
        return pl

class Set(pipeline.Adapter):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __repr__(self):
        return "Set: " + " ".join([kw for kw in self.kwargs])

    def Process(self, pl):
        for kw in self.kwargs:
            if hasattr(pl, kw):
                setattr(pl, kw, self.kwargs[kw])
            else:
                ValueError("Unrecongnized set payload attribute '%s'" %kw)
        self.sayf("{}", self.kwargs)
        return pl


class SetModel(pipeline.Adapter):
    """ Fit a model to the training dataset. """
    def __init__(self, model_instance):
        self.model = model_instance

    def __repr__(self):
        return "SetModel: " + utils.nice_name(self.model)

    def Process(self, pl):
        # fit the model to data
        X = pl.Tr[pl.xCols]
        y = pl.Tr[pl.yCol]
        self.model.fit(X, y)
        pl.model = self.model
        return pl