import numpy as np
import pandas as pd

import pipeline

def Differences(idf):
    return (idf
           # peak shift
           .assign(lp21 = lambda df: df.lspk2 - df.lspk1)
           .assign(lp31 = lambda df: df.lspk3 - df.lspk1)
           .assign(lp32 = lambda df: df.lspk3 - df.lspk2)
           .assign(tp21 = lambda df: df.tspk2 - df.tspk1)
           .assign(tp31 = lambda df: df.tspk3 - df.tspk1)
           .assign(tp32 = lambda df: df.tspk3 - df.tspk2)
           # peak broadening
           .assign(lw21 = lambda df: df.lsfw2 - df.lsfw1)
           .assign(lw31 = lambda df: df.lsfw3 - df.lsfw1)
           .assign(lw32 = lambda df: df.lsfw3 - df.lsfw2)
           .assign(tw21 = lambda df: df.tsfw2 - df.tsfw1)
           .assign(tw31 = lambda df: df.tsfw3 - df.tsfw1)
           .assign(tw32 = lambda df: df.tsfw3 - df.tsfw2)
           # distance between lspr and tspr peaks
           .assign(dp11 = lambda df: df.lspk1 - df.tspk1)
           .assign(dp22 = lambda df: df.lspk2 - df.tspk2)
           .assign(dp33 = lambda df: df.lspk3 - df.tspk3)
           .assign(dp21 = lambda df: df.dp22 - df.dp11)
           .assign(dp31 = lambda df: df.dp33 - df.dp11)
           .assign(dp32 = lambda df: df.dp33 - df.dp22)
           # difference between lspr and tspr bandwidth
           .assign(dw11 = lambda df: df.lsfw1 - df.tsfw1)
           .assign(dw22 = lambda df: df.lsfw2 - df.tsfw2)
           .assign(dw33 = lambda df: df.lsfw3 - df.tsfw3)
           .assign(dw21 = lambda df: df.dw22 - df.dw11)
           .assign(dw31 = lambda df: df.dw33 - df.dw11)
           .assign(dw32 = lambda df: df.dw33 - df.dw22)
    )

def InverseDifferences(idf):
    df = (idf
           # peak shift
           .assign(ilp21 = lambda df: 1.0 / df.lp21)
           .assign(ilp31 = lambda df: 1.0 / df.lp31)
           .assign(ilp32 = lambda df: 1.0 / df.lp32)
           .assign(itp21 = lambda df: 1.0 / df.tp21)
           .assign(itp31 = lambda df: 1.0 / df.tp31)
           .assign(itp32 = lambda df: 1.0 / df.tp32)
           # peak broadening
           .assign(ilw21 = lambda df: 1.0 / df.lw21)
           .assign(ilw31 = lambda df: 1.0 / df.lw31)
           .assign(ilw32 = lambda df: 1.0 / df.lw32)
           .assign(itw21 = lambda df: 1.0 / df.tw21)
           .assign(itw31 = lambda df: 1.0 / df.tw31)
           .assign(itw32 = lambda df: 1.0 / df.tw32)
           # distance between lspr and tspr peaks
           .assign(idp11 = lambda df: 1.0 / df.dp11)
           .assign(idp22 = lambda df: 1.0 / df.dp22)
           .assign(idp33 = lambda df: 1.0 / df.dp33)
           .assign(idp21 = lambda df: 1.0 / df.dp21)
           .assign(idp31 = lambda df: 1.0 / df.dp31)
           .assign(idp32 = lambda df: 1.0 / df.dp32)
           # difference between lspr and tspr bandwidth
           .assign(idw11 = lambda df: 1.0 / df.dw11)
           .assign(idw22 = lambda df: 1.0 / df.dw22)
           .assign(idw33 = lambda df: 1.0 / df.dw33)
           .assign(idw21 = lambda df: 1.0 / df.dw21)
           .assign(idw31 = lambda df: 1.0 / df.dw31)
           .assign(idw32 = lambda df: 1.0 / df.dw32)
           )

    # reset infinite values
    df.replace([np.inf, -np.inf], 10000, inplace=True)
    return df


class AggregateFeatures(pipeline.Adapter):
    """ Calculate difference and inverse difference features. """
    def __init__(self, show=True):
        self.show = show
    def Process(self, pl):
        oldfeats = pl.Tr.columns
        
        pl.Tr = Differences(pl.Tr)
        pl.Tr = InverseDifferences(pl.Tr)
        
        if pl.Tv is not None:
            pl.Tv = Differences(pl.Tv)
            pl.Tv = InverseDifferences(pl.Tv)
            
        pl.Ts = Differences(pl.Ts)
        pl.Ts = InverseDifferences(pl.Ts)
        
        newfeats = pl.Tr.columns.difference(oldfeats)
        if self.show:
            self.sayf("Added {} new features.", len(newfeats))
            self.sayf("{}", list(newfeats))
        return pl
