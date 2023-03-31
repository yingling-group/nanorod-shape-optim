import sys 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

class Regressor:
    def __init__(self, df = None):
        if df is not None:
            assert isinstance(df, pd.DataFrame), "DataFrame needed"
        self.out = df

        # transform handlers
        self.xsclr = None 
        self.ysclr = None 
        self.featFns = []

        # data attributes
        self.xCols = None 
        self.yCol = None
        self.model = None

    def _set_cols(self, xcols, ycol = None, df = None):
        if df is None: df = self.out
        assert isinstance(df, pd.DataFrame), f"df must be a DataFrame, not {type(df)}"
        assert not isinstance(xcols, str), "xcols must be a list"
        for c in xcols: assert c in df.columns, \
            f"Column {c} not in dataframe"

        self.xCols = xcols
        self.xsclr = StandardScaler().fit(df[xcols])
        if ycol is not None:
            assert isinstance(ycol, str), "Only 1 y column supported"
            assert ycol in df.columns, f"{ycol} not in dataframe"
            self.yCol = ycol
            self.ysclr = StandardScaler().fit(df[[ycol]].values)
        self.out = df

    def ScaleX(self, X = None):
        if X is None: X = self.out
        if self.xsclr is None:
            return X

        assert isinstance(X, pd.DataFrame), f"X must be a DataFrame, not {type(X)}"
        assert self.xCols is not None, "No xCols set yet"

        for c in self.xCols:
            assert c in X.columns, f"Column not found: {c}, please call Prep() first"

        df = self.xsclr.transform(X[self.xCols])
        df = pd.DataFrame(df, index=X.index, columns=self.xCols)

        self.out = df
        return df

    def UnscaleX(self, X = None):
        if X is None: X = self.out
        if self.xsclr is None:
            return X

        assert isinstance(X, pd.DataFrame), f"X must be a DataFrame, not {type(X)}"
        assert self.xCols is not None, "No xCols set yet"

        for c in self.xCols:
            assert c in X.columns, f"Column not found: {c}, please call Prep() first"

        df = self.xsclr.inverse_transform(X[self.xCols])
        df = pd.DataFrame(df, index=X.index, columns=self.xCols)

        self.out = df
        return df

    def ScaleY(self, y = None):
        if y is None: y = self.out
        assert isinstance(y, pd.Series), f"Series required to Scale, not {type(y)}"
        assert self.yCol is not None, "No yCol set yet"

        if self.ysclr is None:
            return y

        ya = y.values.reshape(-1, 1)
        ys = pd.Series(self.ysclr.transform(ya).flatten(),
                      name = self.yCol, index=y.index)
        self.out = ys
        return ys

    def UnscaleY(self, y = None):
        if y is None: y = self.out
        assert isinstance(y, pd.Series), f"Series required to Unscale, not {type(y)}"
        assert self.yCol is not None, "No yCol set yet"

        if self.ysclr is None:
            return y

        ya = y.values.reshape(-1, 1)
        ys = pd.Series(self.ysclr.inverse_transform(ya).flatten(),
                      name = self.yCol, index=y.index)

        self.out = ys
        return ys

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
        
        Xdf = pd.concat([df[ocols], self.ScaleX(df)], axis=1)
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
        df = pd.concat([df[ocols], self.UnscaleX(df)], axis=1)
        
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

    def AddFeatures(self, *fnlist, df = None, show_list = True):
        if df is None: df = self.out
        assert isinstance(df, pd.DataFrame), f"df must be a DataFrame, not {type(df)}"

        for fn in fnlist:
            oldfeats = df.columns
            df = fn(df)
            newfeats = df.columns.difference(oldfeats)
            print("%s() added %d new features." %(fn.__name__, len(newfeats)))
            if show_list:
                print(list(newfeats), "\n")
            self.featFns.append(fn)
        self.out = df
        return df
    
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
            

    def FitModel(self, model, Tr = None, Ts = None, plot=True, save=False):
        if Tr is None and Ts is None:
            Tr = self.out[0]
            Ts = self.out[1]
        
        if save:
            assert isinstance(save, str), "save should be a file name"

        # Model fitting
        name = str(model).split("(")[0]
        assert self.model is None, "Cannot refit the same class, please create a new one."

        print(f"Fitting {self.yCol}={name}() ...", end =" ")
        sys.stdout.flush()

        model.fit(
            self.ScaleX(Tr[self.xCols]),
            self.ScaleY(Tr[self.yCol])
        )
        self.model = model

        print("OK.")
        if plot:
            return self._model_fitness(Tr, Ts, save)

        self.out = Tr, Ts
        return self

    def Fit(self, xcols, ycol, model, **kwargs):
        self.Split(xcols, ycol)
        self.FitModel(model, **kwargs)
        return self

    def Predict(self, df = None, scaleX=True, unscaleY=True):
        if df is None: df = self.out
        assert isinstance(df, pd.DataFrame), f"df must be a DataFrame, not {type(df)}"
        assert self.model != None, "Please train a model first"
        for c in self.xCols:
            assert c in df.columns, \
                f"Column not found: {c}, please call Prep() first"

        X = df[self.xCols]
        if scaleX:
            X = self.ScaleX(X)

        yp = self.model.predict(X)
        yp = pd.Series(yp, name=self.yCol, index = X.index)
        if unscaleY:
            yp = self.UnscaleY(yp)

        self.out = yp
        return yp
    
    @property
    def Name(self):
        if self.model is not None:
            return str(self.model).split("(")[0]

    def __repr__(self):
        desc = str(self.__class__).split("'")[1] + " [\n\t"

        if self.yCol is not None:
            desc += self.yCol + " = "
        if self.model is not None:
            desc += self.Name[0:3] + f"({self.xCols})"
            
        desc += "\n]"
        return desc

    def _model_fitness(self, Tr, Ts, save):
        xtr = Tr[self.xCols]
        xts = Ts[self.xCols]
        ytr = Tr[self.yCol]
        yts = Ts[self.yCol]
        ptr = self.Predict(xtr)
        pts = self.Predict(xts)

        R2 = r2_score(ytr, ptr)
        MSE = mean_squared_error(ytr, ptr)

        # Plot
        fig, ax = plt.subplots(1, 3, figsize=(8, 2.25))
        ax[0].plot(ytr, ptr, 'bx')
        dline = np.linspace(*ax[0].get_xlim())
        ax[0].plot(dline, dline, 'k--')
        ax[0].set_xlabel("Prior predictor")
        ax[0].set_title("R2 = %0.2f, RMSE = %0.2f" %(R2, np.sqrt(MSE)))
        
        R2 = r2_score(yts, pts)
        MSE = mean_squared_error(yts, pts)

        ax[1].plot(yts, pts, 'bx')
        dline = np.linspace(*ax[1].get_xlim())
        ax[1].plot(dline, dline, 'k--')
        ax[1].set_xlabel("Posterior predictor")
        ax[1].set_title("R2 = %0.2f, RMSE = %0.2f" %(R2, np.sqrt(MSE)))
        
        ax[2].axhline(y=0, linestyle='--')
        ax[2].plot(yts, pts - yts, 'r.')
        ax[2].set_xlabel("Residuals")

        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=600)
            plt.close(fig)
        else:
            plt.show()
            
        return R2, MSE

    def Accuracy(self, df):
        assert isinstance(df, pd.DataFrame), f"df must be a DataFrame, not {type(df)}"

        # Add required features
        for fn in self.featFns:
            df = fn(df)

        assert self.yCol in df.columns, "df does not contain the response"

        # Scale
        y = df[self.yCol]
        p = self.Predict(df)

        R2 = r2_score(y, p)
        MSE = mean_squared_error(y, p)

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(5, 2.25), sharey=False)
        ax[0].plot(y, p, 'bx')
        dline = np.linspace(*ax[0].get_xlim())
        ax[0].plot(dline, dline, 'k--')
        ax[0].set_xlabel("True")
        ax[0].set_ylabel("Prediction")
        ax[0].set_title("R2 = %0.2f, RMSE = %0.2f" %(R2, np.sqrt(MSE)))

        ax[1].plot(y, p - y, 'r.')
        ax[1].axhline(y=0, linestyle='--')
        ax[1].set_xlabel("Residuals")

        plt.tight_layout()
        plt.show()
        return R2, MSE

    def Prep(self, df):
        # a dictionary containtaining a single row
        if isinstance(df, dict):
            for c in self.xCols: assert c in df.keys(), f"{c} not in df"
            df = pd.DataFrame(df, index=[0])
        elif isinstance(df, np.ndarray):
            try:
                ncol = df.shape[1]
            except:
                ncol = df.shape[0]
            assert ncol == len(self.xCols), \
                "Not all columns found in array, columns in exact order needed"
            try:
                df = pd.DataFrame(df, columns=self.xCols)
            except:
                df = pd.DataFrame(df, columns=self.xCols, index=[0])

        assert isinstance(df, pd.DataFrame), "DataFrame/ndarray/dict required"

        for c in self.xCols:
            if c not in df.columns:
                # add additional features
                for fn in self.featFns:
                    df = fn(df)
                break

        self.out = df
        return df
