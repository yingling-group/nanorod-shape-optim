import sys 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

class Regressor:
    """ A regressor class to handle pandas DataFrames and maintain the train/test pipeline """

    def __init__(self, df = None):
        if df is not None:
            assert isinstance(df, pd.DataFrame), "DataFrame needed for the training dataset"

        self.Tr = df

        # transform handlers
        self.xsclr = None 
        self.ysclr = None 
        self.featFns = []

        # data attributes
        self.xCols = None 
        self.yCol = None
        self.model = None

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
            self.ysclr = StandardScaler().fit(self.Tr[[ycol]].values)

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
            print("%s() added %d new features." %(fn.__name__, len(newfeats)))
            if show_list:
                print(list(newfeats), "\n")
            self.featFns.append(fn)

    def _prep_df(self, Ts):
        """ Prepare a dataframe/dict/numpy array for prediction by adding features """

        # a dictionary containing a single row
        if isinstance(Ts, dict):
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

    def FitModel(self, model):
        name = str(model).split("(")[0]
        assert self.model is None, "Cannot refit the same Regressor, please create a new one."

        print(f"Fitting {self.yCol} = {name}()", end =" ... ")
        sys.stdout.flush()

        model.fit(
            self._scaleX(self.Tr[self.xCols]),
            self._scaleY(self.Tr[self.yCol])
        )
        self.model = model
        print("OK")

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

    def Augment(self, cols, df = None, K = 3, qcol = "quality", scale = 0.2,
                                            plot = True, save = False):
        if df is None: df = self.out
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

    def Split(self, xcols, ycol, df = None, test_size=0.33, fold=0, K=None):
        if df is None: df = self.out
        assert isinstance(df, pd.DataFrame), f"df must be a DataFrame, not {type(df)}"

        self._set_cols(xcols, ycol, df)
        X = df[xcols]
        y = df[ycol]

        if K is None:
            # normal train test split
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
            
        self.out = Tr, Ts
        return Tr, Ts

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

    def Fitness(self, Ts, save = False):
        """ Calculate and plot fitness of the given test dataset """
        Ts = self._prep_df(Ts)
        xtr = self.Tr[self.xCols]
        xts = Ts[self.xCols]
        ytr = self.Tr[self.yCol]
        yts = Ts[self.yCol]

        # fitness of the training data
        ptr = self.Predict(xtr)

        R2 = r2_score(ytr, ptr)
        MSE = mean_squared_error(ytr, ptr)

        # Plot
        fig, ax = plt.subplots(1, 3, figsize=(8, 2.5))
        ax[0].plot(ytr, ptr, 'bx')
        dline = np.linspace(*ax[0].get_xlim())
        ax[0].plot(dline, dline, 'k--')
        ax[0].set_xlabel("Prior predictor")
        ax[0].set_title("R2 = %0.2f, RMSE = %0.2f" %(R2, np.sqrt(MSE)))

        # fitness of the test data
        pts = self.Predict(xts)

        R2 = r2_score(yts, pts)
        MSE = mean_squared_error(yts, pts)

        ax[1].plot(yts, pts, 'bx')
        dline = np.linspace(*ax[1].get_xlim())
        ax[1].plot(dline, dline, 'k--')
        ax[1].set_xlabel("Posterior predictor")
        ax[1].set_title("R2 = %0.2f, RMSE = %0.2f" %(R2, np.sqrt(MSE)))
        
        ax[2].axhline(y=0, linestyle='--')
        ax[2].plot(yts, pts - yts, 'r.')
        ax[2].set_xlabel("Posterior residuals")

        plt.suptitle(self.Name)
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=600)
            plt.close(fig)
        else:
            plt.show()
            
        return {
            'model': self.Name,
            'R2': R2,
            'MSE': MSE
        }

    def ParityAndResidual(self, Ts):
        """ Plot parity and residuals of the given test dataset """

        assert isinstance(Ts, pd.DataFrame), f"Ts must be a DataFrame, not {type(Ts)}"
        assert self.yCol in Ts.columns, "Ts does not contain the response"

        Ts = self._prep_df(Ts)

        # Scale
        y = Ts[self.yCol]
        p = self.Predict(Ts)

        R2 = r2_score(y, p)
        MSE = mean_squared_error(y, p)

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(5, 2.5), sharey=False)
        ax[0].plot(y, p, 'bx')
        dline = np.linspace(*ax[0].get_xlim())
        ax[0].plot(dline, dline, 'k--')
        ax[0].set_xlabel("True")
        ax[0].set_ylabel("Prediction")
        ax[0].set_title("R2 = %0.2f, RMSE = %0.2f" %(R2, np.sqrt(MSE)))

        ax[1].plot(y, p - y, 'r.')
        ax[1].axhline(y=0, linestyle='--')
        ax[1].set_xlabel("Residuals")

        plt.suptitle(self.Name)
        plt.tight_layout()
        plt.show()
        return {
            'model': self.Name,
            'R2': R2,
            'MSE': MSE
        }


def New(df, xcols, ycol, fnlist):
    reg = Regressor(df)
    reg.SetColumns(xcols, ycol)
    reg.AddFeatures(fnlist)
    return reg
