import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from . import pipeline
from . import utils


class ScaleX(pipeline.Adapter):
    """ Standard scale the xCols of the datasets. """
    def __init__(self, scaler=StandardScaler, allColumns = False):
        self.scaler = scaler
        self.allCols = allColumns
        
    def __repr__(self):
        return "ScaleX: " + utils.nice_name(self.scaler) + " All: " + str(self.allCols)

    def Process(self, pl):
        if self.allCols:
            tCols = pl.Tr.select_dtypes(include=np.number).columns
            if pl.yCol in tCols:
                tCols = tCols.drop(pl.yCol)
            self.sayf("Scaled all columns.")
        else:
            tCols = pl.xCols
            self.sayf("Scaled xCols, non xCols are unchanged.")

        # fit on the training set only
        pl.xsclr = self.scaler().fit(pl.Tr[tCols])
        
        # transform training
        dft = pl.xsclr.transform(pl.Tr[tCols])
        dft = pd.DataFrame(dft, index=pl.Tr.index, columns=tCols)
        pl.Tr.update(dft)
        
        # transform test
        dft = pl.xsclr.transform(pl.Ts[tCols])
        dft = pd.DataFrame(dft, index=pl.Ts.index, columns=tCols)
        pl.Ts.update(dft)

        # transform validation
        if pl.Tv is not None:
            dft = pl.xsclr.transform(pl.Tv[tCols])
            dft = pd.DataFrame(dft, index=pl.Tv.index, columns=tCols)
            pl.Tv.update(dft)
        
        return pl

class UnscaleX(pipeline.Adapter):
    """ Unscale the xCols of the training dataset. """
    def __init__(self, allColumns = False):
        self.allCols = allColumns
    def Process(self, pl):
        if pl.xsclr is None: return pl
        if self.allCols:
            tCols = pl.Tr.select_dtypes(include=np.number).columns
            if pl.yCol in tCols:
                tCols = tCols.drop(pl.yCol)

        else:
            tCols = pl.xCols
        
        dft = pl.xsclr.inverse_transform(pl.Tr[tCols])
        dft = pd.DataFrame(dft, index=pl.Tr.index, columns=tCols)
        pl.Tr.update(dft)

        dft = pl.xsclr.inverse_transform(pl.Ts[tCols])
        dft = pd.DataFrame(dft, index=pl.Ts.index, columns=tCols)
        pl.Ts.update(dft)

        if pl.Tv is not None:
            dft = pl.xsclr.inverse_transform(pl.Tv[tCols])
            dft = pd.DataFrame(dft, index=pl.Tv.index, columns=tCols)
            pl.Tv.update(dft)

        return pl
