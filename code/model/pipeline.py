import warnings
import sys
import numpy as np
import itertools
from typing import Iterable

import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import warnings

def find_noncollinear_features(df, x0, columns, vif_cutoff, return_failed = False):
    """ Try to add columns one by one to dataframe without creating multicollinearity.
        Return the list of features that passed the VIF check.
        Use standard scaled values to reduce VIF.
    """

    candidate = []
    collinear = []
    
    # Python passes lists by reference
    passed = x0.copy()
    
    for col in columns:
        if col in passed:
            continue
        try:
            vif = calc_vif(df[passed + [col]]).drop(col, axis=1)
        except:
            continue
        if vif.iloc[0, 0] < vif_cutoff:
            passed.append(col)
        else:
            candidate.append(col)
            collinear.append(vif.columns[0])

    failed = pd.DataFrame({'linearColumn': collinear}, index=candidate)

    # vif table, table of failed features
    if return_failed:
        return calc_vif(df[passed]), failed
    else:
        return calc_vif(df[passed])


class Pipeline:
    """ A class to handle pandas DataFrames and maintain the train/test pipeline """

    def __init__(self, df = None):
        if df is not None:
            assert isinstance(df, pd.DataFrame), "DataFrame needed for the training dataset"

        self.Tr = df
        self.Ts = None

        # transform handlers
        self.xsclr = None 
        self.ysclr = None 
        self.featFns = []

        # data attributes
        self.xCols = None 
        self.yCol = None
        self.model = None

    def NonCollinearFeatures(self, keepCols = [], ignoreCols = [], vif_cutoff = 8):
        """ Return the best features that do not create multicollinearity.
            @Todo: remove this method.
        """
        df = self.Tr.drop(ignoreCols, axis=1)
        sclr = StandardScaler().fit(df)
        sdf = pd.DataFrame(sclr.transform(df), index=df.index, columns=df.columns)
        return find_noncollinear_features(sdf, keepCols, df.columns, vif_cutoff)

    def SetColumns(self, xcols, ycol = None):
        """ Specify the x and y columns of the training dataset """
        assert not isinstance(xcols, str), "xcols must be a list"
        for c in xcols: assert c in self.Tr.columns, \
            f"Column {c} not in dataframe"

        self.xCols = xcols
        self.xsclr = StandardScaler().fit(self.Tr[xcols])
        if ycol is not None:
            assert isinstance(ycol, str), "Only 1 y column supported"
            assert ycol in self.Tr.columns, f"{ycol} not in dataframe"
            self.yCol = ycol
            try:
                self.ysclr = StandardScaler().fit(self.Tr[[ycol]].values)
            except ValueError:
                # print("Warning -- Y standard scaling failed.")
                pass

    def _scaleX(self, X):
        """ Scale a given dataframe of features """

        assert isinstance(X, pd.DataFrame), f"X must be a DataFrame, not {type(X)}"
        assert self.xCols is not None, "No xCols set yet, please call SetColumns() first"

        if self.xsclr is None:
            return X
        else:
            df = self.xsclr.transform(X[self.xCols])
            df = pd.DataFrame(df, index=X.index, columns=self.xCols)
            return df

    def _unscaleX(self, X):
        """ UnScale a given dataframe of features """

        assert isinstance(X, pd.DataFrame), f"X must be a DataFrame, not {type(X)}"
        assert self.xCols is not None, "No xCols set yet, please call SetColumns() first"

        if self.xsclr is None:
            return X
        else:
            df = self.xsclr.inverse_transform(X[self.xCols])
            df = pd.DataFrame(df, index=X.index, columns=self.xCols)
            return df

    def _scaleY(self, y):
        """ Scale a given y Series """

        assert isinstance(y, pd.Series), f"Series required to Scale, not {type(y)}"
        assert self.yCol is not None, "No yCol set yet, please call SetColumns() first"

        if self.ysclr is None:
            return y
        else:
            ya = y.values.reshape(-1, 1)
            ys = pd.Series(self.ysclr.transform(ya).flatten(),
                        name = self.yCol, index=y.index)
            return ys

    def _unscaleY(self, y):
        """ UnScale a given y Series """

        assert isinstance(y, pd.Series), f"Series required to Unscale, not {type(y)}"
        assert self.yCol is not None, "No yCol set yet, please call SetColumns() first"

        if self.ysclr is None:
            return y
        else:
            ya = y.values.reshape(-1, 1)
            ys = pd.Series(self.ysclr.inverse_transform(ya).flatten(),
                        name = self.yCol, index=y.index)
            return ys

    def AddFeatures(self, fnlist, show_list = True):
        """ Add additional features to the training dataset """

        for fn in fnlist:
            oldfeats = self.Tr.columns
            self.Tr = fn(self.Tr)
            newfeats = self.Tr.columns.difference(oldfeats)
            if show_list:
                print("%s() added %d new features." %(fn.__name__, len(newfeats)))
                print(list(newfeats), "\n")
            self.featFns.append(fn)

    def FitModel(self, model):
        name = str(model).split("(")[0]
        assert self.model is None, "Cannot refit the model, please create a new one."

        print(f"Fitting {self.yCol} = {name}()", end =" ... ")
        sys.stdout.flush()

        model.fit(
            self._scaleX(self.Tr[self.xCols]),
            self._scaleY(self.Tr[self.yCol])
        )
        self.model = model
        print("OK")

    def _prep_df(self, Ts = None):
        """ Prepare a dataframe/dict/numpy array for prediction by adding features.
            If no dataset is given, use the previously splitted test dataset.
        """
        if Ts is None:
            Ts = self.Ts

        # a dictionary containing a single row
        elif isinstance(Ts, dict):
            for c in self.xCols: assert c in Ts.keys(), f"{c} not in Ts"
            Ts = pd.DataFrame(Ts, index=[0])

        # a numpy array
        elif isinstance(Ts, np.ndarray):
            try:
                ncol = Ts.shape[1]
            except:
                ncol = Ts.shape[0]
            assert ncol == len(self.xCols), \
                "Not all columns found in array, columns in exact order needed"
            try:
                Ts = pd.DataFrame(Ts, columns=self.xCols)
            except:
                Ts = pd.DataFrame(Ts, columns=self.xCols, index=[0])

        # pandas dataframe
        assert isinstance(Ts, pd.DataFrame), "DataFrame/ndarray/dict required"

        # add additional features
        for fn in self.featFns:
            Ts = fn(Ts)

        return Ts

    def Predict(self, df):
        assert isinstance(df, pd.DataFrame), f"df must be a DataFrame, not {type(df)}"
        assert self.model != None, "Please train a model first"
        for c in self.xCols:
            assert c in df.columns, \
                f"Column not found: {c}, please call PrepPrediction() first"

        X = df[self.xCols]
        X = self._scaleX(X)

        yp = self.model.predict(X)
        yp = pd.Series(yp, name=self.yCol, index = X.index)
        yp = self._unscaleY(yp)

        return yp
    
    def HypParamSearch(self, hyperparams, scoring, grid = False, cv = 5, Ts = None):
        """ Follow the same style as GridSearchCV. Give a test dataset
            to include it in the CV process, else use the training set only.
        """
        if grid:
            print("Running grid search", end = " ... ")
            clf = GridSearchCV(self.model, hyperparams, cv=cv,
                               n_jobs = -1,
                               scoring = scoring,
                              )
        else:
            print("Running randomized search", end = " ... ")
            clf = RandomizedSearchCV(self.model, hyperparams, cv = cv,
                                     n_iter = 100,
                                     n_jobs = -1,
                                     scoring = scoring,
                                    )

        X = self._scaleX(self.Tr)
        y = self._scaleY(self.Tr[self.yCol])
        if Ts is not None:
            Ts = self._prep_df(Ts)
            xts = self._scaleX(Ts)
            yts = self._scaleY(Ts[self.yCol])
            X = pd.concat([X, xts])
            y = pd.concat([y, yts])

        best = clf.fit(X, y)
        print("OK")
        print("best score:", best.best_score_)
        return best

    def FeatureSearch(self, scoring, direction, cv = 5, Ts = None, tol=None):
        """ Find the best features using SequentialFeatureSelector.
            Give a test dataset to include it in the CV process,
            else use the training set only.
        """
        print("Running features search", end = " ... ")
        clf = SequentialFeatureSelector(self.model,
                                        direction = direction,
                                        cv=cv,
                                        n_jobs = -1,
                                        n_features_to_select = 15 if tol is None else 'auto',
                                        tol = tol,
                                        scoring = scoring)

        X = self._scaleX(self.Tr)
        y = self._scaleY(self.Tr[self.yCol])
        if Ts is not None:
            Ts = self._prep_df(Ts)
            xts = self._scaleX(Ts)
            yts = self._scaleY(Ts[self.yCol])
            X = pd.concat([X, xts])
            y = pd.concat([y, yts])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best = clf.fit(X, y)
        print("OK")
        print("best features:", list(X.columns[best.support_]))
        return X.columns[best.support_]

    def Augment(self, cols, df, K = 3, qcol = "quality", scale = 0.2,
                                            plot = True, save = False):
        assert K <= 5, "Too high value of K will make dataset invalid"
        if save:
            assert isinstance(save, str), "Please provide a filename."
        assert isinstance(df, pd.DataFrame), f"df must be a DataFrame, not {type(df)}"
        
        # other columns to be kept constant
        ocols = list(df.columns[~df.columns.isin(cols)])

        self._set_cols(cols, None, df=df)
        
        Xdf = pd.concat([df[ocols], self._scaleX(df)], axis=1)
        Xdf.index = np.arange(1, len(Xdf) + 1)
        Xdf = Xdf.assign(augmented = False)
        df = Xdf.copy()

        N = 0
        for i in range(Xdf.shape[0]):
            row = Xdf.iloc[i:i+1]
            row = row.assign(augmented = True)
            q = row.iloc[0,][qcol]
            stdev = scale / q
            rep = int(q ** K / 5)

            for j in range(1, rep):
                row.index = [len(Xdf) + N + 1]
                sample = pd.DataFrame(np.random.normal(loc = row[cols],
                                                       scale= stdev),
                                      columns = cols, index = row.index)
                nrow = pd.concat([row[ocols], sample], axis=1)
                df = pd.concat([df, nrow], axis=0)
                N = N + 1
                if N % 100 == 0: print("\rAdded %d new rows" %N, end ="")

        print("\rOK. Added %d new rows..." %N)
        
        # Unscale Back
        df = pd.concat([df[ocols], self._unscaleX(df)], axis=1)
        
        # Shuffle
        df = df.sample(frac=1)

        # Diagnostics
        if plot:
            fig, ax = plt.subplots(figsize=(8, 3))
            df[qcol].hist(bins=20, ax = ax)
            # qcol should be within 1 to 10
            ax.set_xticks(np.arange(1, 11))
            ax.set(xlabel = qcol, ylabel = "Count", title="Augmented Dataset")
            plt.tight_layout()
            if save:
                plt.savefig(save, dpi=600)
                plt.close(fig)
            else:
                plt.show()
                
        self.out = df
        return df

    def Compare(self, df1, df2 = None, scaled=False):
        if df2 is None: df2 = self.out

        assert isinstance(df1, pd.DataFrame), "DataFrame needed"
        assert isinstance(df2, pd.DataFrame), f"df must be a DataFrame, not {type(df2)}"

        xcols = np.intersect1d(df1.columns, df2.columns, True)

        if scaled:
            sclr = StandardScaler().fit(df1[xcols])
            df1 = pd.DataFrame(sclr.transform(df1[xcols]), columns=xcols)
            df2 = pd.DataFrame(sclr.transform(df2[xcols]), columns=xcols)

        for i in range(len(xcols)):
            col1 = xcols[i]
            for j in range(i+1, len(xcols)):
                col2 = xcols[j]
                fig, ax = plt.subplots(1, 1, sharey=True, dpi=80,
                                       figsize=(2.5, 2.5))
                ax.plot(df2[col1], df2[col2], 'r.', alpha=0.7, label="df2")
                ax.plot(df1[col1], df1[col2], 'k.', label="df1")
                ax.set(xlabel = col1, ylabel=col2)
                ax.legend()
                # plt.tight_layout()
                plt.show()

    def Split(self, test_size=0.20, fold=0, K=None):
        """ Perform stratification of the dataset.
            If K is specified, select the fold-th fold as test set.
        """
        df = self.Tr.copy()

        X = df[self.xCols]
        y = df[self.yCol]

        if K is None:
            # Normal train test split of scikit-learn
            xtr, xts, ytr, yts = train_test_split(X, y, test_size=test_size)
            print(f"Selected random {test_size} of rows as test set.")
            Tr = pd.concat([xtr, ytr], axis=1)
            Ts = pd.concat([xts, yts], axis=1)
        else:
            # K fold split
            assert fold < K, f"Fold number should be between 0 to K-1"
            print(f"Selected fold {fold} as test set.")
            nrows = df.shape[0]
            foldsize = nrows // K 
            start = fold * foldsize
            end = start + foldsize
            Ts = df.iloc[start:end]
            Tr = pd.concat((df.iloc[0:start],
                            df.iloc[end:nrows]), axis=0)
            Tr = Tr.sample(frac=1)
            
        self.Tr = Tr 
        self.Ts = Ts

        print("\t %d rows for testing" %Ts.shape[0])
        print("\t %d rows for training" %Tr.shape[0])

    @property
    def Name(self):
        if self.model is not None:
            return str(self.model).split("(")[0]
        else:
            return "<Not trained>"

    def __repr__(self):
        desc = str(self.__class__).split("'")[1] + " [\n\t"

        if self.yCol is not None:
            desc += self.yCol + " = "
        if self.model is not None:
            desc += self.Name[0:3] + f"({self.xCols})"
            
        desc += "\n]"
        return desc


class Payload:
    """ Payload to pass between the Adapters of a GridLine """
    def __init__(self, **kwargs):
        self.Tr = None
        self.Ts = None
        self.xCols = None 
        self.yCol = None
        self.xsclr = None 
        self.ysclr = None

    def __repr__(self):
        return "Tr Head:\n" + repr(self.Tr.head())

class Adapter:
    """ The Adapter class to be overridden by each GridLine item. """
    def NewLine(self, pl):
        # Called on each new pipeline
        # Override if needed, make sure to call super()
        self.output = ""

    def __repr__(self):
        pkg = str(self.__class__).split("'")[1]
        if "." in pkg:
            pkg = pkg.split(".")[-1]
        return pkg + "()"

    def sayf(self, msg, *args, **kwargs):
        eol = "\n"
        if 'end' in kwargs:
            eol = kwargs['end']
        self.output += msg.format(*args) + eol

    def _report(self):
        if len(self.output) == 0:
            return None
        else:
            out = "\n" + self.output
            out = out.replace("\n", "\n\t ")
            return out.rstrip()

    def Process(self, X):
        pass


class GridLine:
    """ Define a list of Adapters, whose Process() methods will be
        called one after another with previous result being passed on. """
    def __init__(self, grid):
        self.grid = []
        self.results = []
        for item in grid:
            if isinstance(item, Iterable):
                self.grid.append(item)
            else:
                self.grid.append([item])
        self.grid = itertools.product(*self.grid)

    def _pipeline(self, i, pipe, X):
        for adapter in pipe:
            if adapter is None:
                continue
            assert isinstance(adapter, Adapter)
            print(" --", adapter, end = " ... ")
            adapter.NewLine(X)
            X = adapter.Process(X)
            rep = adapter._report()
            if rep:
                print(rep, end="\n\n")
            else:
                print("ok")

        # final payload
        return X

    def Execute(self, X):
        self.results = []
        for i, pipe in enumerate(self.grid):
            print("Pipeline %02d:" %(i+1))
            if hasattr(X, 'copy'):
                xclone = X.copy()
            else:
                xclone = X
            res = self._pipeline(i+1, pipe, xclone)
            self.results.append(res)
            print("Done %02d.\n" %(i+1))
