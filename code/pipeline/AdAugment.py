import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from . import pipeline
from . import utils

class AugmentImb(pipeline.Adapter):
    """ Augment the training dataset using one of the `imblearn` methods. """
    def __init__(self, sampler_instance, show=True):
        self.overSampler = sampler_instance
        self.show = show

    def __repr__(self):
        return "AugmentImb: " + utils.nice_name(self.overSampler)

    def Process(self, pl):
        shp = pl.Tr.shape
        X = pl.Tr.drop(pl.yCol, axis=1)
        y = pl.Tr[pl.yCol]
        Xs, ys = self.overSampler.fit_resample(X, y)
        pl.Tr = pd.concat([Xs, ys], axis=1)
        self.sayf("Old shape: {}, New shape: {}", shp, pl.Tr.shape)
        return pl


class AugmentByQuality(pipeline.Adapter):
    """ Augment the training dataset by a quality factor.
        F : Frequency parameter, higher = more rows added
        scale : Perturbation parameter, lower = sample more similar to originals
    """
    def __init__(self, F = 2, scale = 0.2, qcol = "quality"):
        assert F <= 5, "Too high value of K will make a dataset invalid"
        self.f = F
        self.qcol = qcol
        self.scl = scale

    def __repr__(self):
        return "AugmentByQuality: F=%d scale=%.2f" %(self.f, self.scl)

    def Process(self, pl):
        shp = pl.Tr.shape

        assert pl.xCols is not None
        df = pl.Tr[pl.xCols] # data subset to augment
        
        sclr = StandardScaler().fit(df)
        dft = sclr.transform(df) # perform augmentation in the scaled range
        
        C = pl.Tr.shape[0] # current row count
        N = 0 # count of newly added rows
        for i in range(pl.Tr.shape[0]):
            # for each existing row
            row = pl.Tr.iloc[i:i+1].copy()
            q = row.iloc[0,][self.qcol] # the quality value
            stdev = self.scl / q # higher quality = lower stdev.
            rep = int(q ** self.f / 5) # higher quality = more frequent
            
            for j in range(1, rep+1):
                # sample around this row `rep` times
                N += 1
                sample = np.random.normal(loc=dft[i:i+1, :], scale=stdev)
                sample = sclr.inverse_transform(sample) # unscale new values
                sdf = pd.DataFrame(sample,
                                   columns = df.columns,
                                   index = [C + N])
                row.index = [C + N] # needed for alignment
                row.update(sdf) # update the old row with new values
                pl.Tr = pd.concat([pl.Tr, row], axis=0) # append row
                
        self.sayf("Old shape: {}, New shape: {}", shp, pl.Tr.shape)
        return pl



class PlotFrequency(pipeline.Adapter):
    """ Plot count/frequency of available data by a column. """
    def __init__(self, col, saveAs=None, once=False):
        self.saveAs = saveAs
        self.col = col
        self.plotOnce = once
        self._plotted = False

    def _plot(self, df):
        fig, ax = plt.subplots(figsize=(3.25, 2.2))
        df[self.col].hist(bins=20, ax = ax, label = "N = %d" %df.shape[0])
        # ax.set_xticks(np.arange(df[self.qcol].min(),
        #                         df[self.qcol].max()+1))
        ax.set(xlabel = self.col, ylabel = "Count")
        ax.grid(False)
        ax.legend()
        if self.saveAs:
            plt.savefig(self.saveAs)
        plt.show()

    def Process(self, pl):
        if self.plotOnce and self.plotted:
            return pl
        else:
            self._plot(pl.Tr)
            self._plotted = True
            return pl

class PlotCount(pipeline.Adapter):
    """ Plot count of available data by a column. """
    def __init__(self, col, xtickLabels = [], saveAs=None, once=False):
        self.saveAs = saveAs
        self.col = col
        self.xtickLabels = xtickLabels
        self.plotOnce = once
        self._plotted = False

    def _plot(self, df):
        fig, ax = plt.subplots(figsize=(3.25, 2.2))
        sns.countplot(x=df[self.col], ax = ax)
        plt.minorticks_off()
        ax.set(ylabel = "Count", xlabel = "")
        
        if len(self.xtickLabels) > 0:
            ax.set(xticklabels = self.xtickLabels)

        if self.saveAs:
            plt.savefig(self.saveAs)
        plt.show()

    def Process(self, pl):
        if self.plotOnce and self.plotted:
            return pl
        else:
            self._plot(pl.Tr)
            self._plotted = True
            return pl


class PlotPerturbation(pipeline.Adapter):
    """ Make scatterplots of two dataframes to compare augmentation. """
    def __init__(self, orig, xCol = None, yCols = [], saveAs=None, scaled=False):
        self.saveAs = saveAs
        self.orig = orig
        self.scld = scaled
        self.xCol = xCol
        self.yCols = yCols

    def _plot(self, df):
        df1 = self.orig.select_dtypes(np.number)
        df2 = df.select_dtypes(np.number)
        # choose the common columns
        xcols = np.intersect1d(df1.columns, df2.columns, True)

        if self.scld:
            sclr = StandardScaler().fit(df1[xcols])
            df1 = pd.DataFrame(sclr.transform(df1[xcols]), columns=xcols)
            df2 = pd.DataFrame(sclr.transform(df2[xcols]), columns=xcols)

        n = 0
        count = len(xcols)
        fig, axes = plt.subplots(2, 2, figsize=(3.5, 3), sharex=True)
        axes2 = axes.flatten()

        col1 = self.xCol

        for col2 in self.yCols:
            ax = axes2[n]
            ax.plot(df2[col1], df2[col2], 'r.', alpha=0.7, label="new")
            ax.plot(df1[col1], df1[col2], 'kx', label="orig")
            ax.set(ylabel=col2)

            n += 1
            if n > 20:
                break

        for i in range(2):
            axes[-1, i].set(xlabel=col1)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1, wspace=0.45)

        if self.saveAs:
            plt.savefig(self.saveAs)
        
        plt.show()

    def Process(self, pl):
        self._plot(pl.Tr)
        return pl
