import os
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.metrics import mean_squared_error

# -----------------------------------------------------------------------------
class TwoLevelsDesign:
    def __init__(self, model, maxInteraction = 2, columns = None):
        self.table = None
        self.effects = []
        self.responses = []
        
        assert model.Xtr is not None, "Please split the data set first"
        
        if columns is None:
            columns = model.Xtr.columns

        params = {}
        for x in zip(columns, model.Xtr[columns].min(), model.Xtr[columns].max()):
            # name : (min, max)
            params[x[0]] = (x[1], x[2])
        
        self.factors = params

        assert len(self.factors) <= 20, "Too many factors! Even 20 factors will need a million evaluations."

        self.createFactorial(self.factors)
        self.interacts = self.createInteraction(maxInteraction)

        print(f"Total {self.table.shape[0]} experiments to run.")

    def createFactorial(self, factors):
        levels = len(factors) * [(-1, +1)]
        self.table = pd.DataFrame([i for i in itertools.product(*levels)])
        self.table.columns = factors.keys()

    def createInteraction(self, maxInteraction):
        sep = "*"
        intnames = []
        for i in range(2, self.table.shape[1] + 1):
            if i > maxInteraction:
                break
            items = [j for j in itertools.combinations(self.table.columns, i)]
            for j in items:
                intnames.append(sep.join(j))

        interactions = {}
        for i in intnames:
            cols = i.split(sep)
            interactions[i] = self.table[cols].product(axis=1).values
            
        return pd.DataFrame(interactions, columns = intnames)

    @property
    def shape(self):
        return self.table.shape[0], self.table.shape[1] + self.interacts.shape[1]
    
    @property
    def columns(self):
        return list(self.table.columns) + list(self.interacts.columns.values)
    
    def __repr__(self):
        table = self.table.copy()
        if self.interacts is not None:
            table = pd.concat([table, self.interacts], axis=1)
        return str(table)

    def run(self, model, *fns, trace=False):
        '''
        fn = function to call with different variables levels.
        K = number of replications
        '''
        runtable = self.table.copy()
        for col in runtable:
            runtable[col] = runtable[col].replace(-1, self.factors[col][0])
            runtable[col] = runtable[col].replace(1, self.factors[col][1])
            
        K = len(fns)
        for cv, fn in enumerate(fns):
            # shuffle to randomize the expts
            runtable = runtable.sample(frac=1)
            results = {}
            for index, row in runtable.iterrows():
                if trace: print(f"\n[{cv+1}_{index:02d}] {fn.__name__}(**{runtable.columns})")
                # import pdb; pdb.set_trace()
                res = fn(model, row.to_frame().T)
                if trace: print(f"[{cv+1}_{index:02d}] = {res}")
                
                try:
                    # vector response
                    for i in range(len(res)):
                        col = "y%d" %i
                        if col not in results:
                            results[col] = [res[i]]
                        else:
                            results[col].append(res[i])

                except TypeError:
                    # scaler response
                    if 'y' not in results:
                        results['y'] = [res]
                    else:
                        results['y'].append(res)

            resTable = pd.DataFrame(results, index=runtable.index)
            self.responses.append(resTable.sort_index())
            if K > 1: print("------------- K = %d OK" %(cv+1))
        self.calcEffects()

    def calcEffects(self):
        for resTable in self.responses:
            table = pd.concat([self.table, self.interacts], axis=1)
            table.reindex(resTable.index)
            # number of levels = 2
            den = table.shape[0] // 2
            eff = table.T.dot(resTable) / den
            self.effects.append(eff)
    
    def TTest(self):
        a = pd.concat(self.effects, axis=1)
        res = stats.ttest_1samp(a, 0, axis=1)
        return pd.DataFrame({
            'statistic': res.statistic,
            'pvalue': res.pvalue
        }, index = a.index)

    def Avg(self):
        a = pd.concat(self.effects)
        g = a.groupby(a.index)
        g = g.mean()
        return g.reindex(self.effects[0].index)
    
    def Std(self):
        a = pd.concat(self.effects)
        g = a.groupby(a.index)
        g = g.std()
        return g.reindex(self.effects[0].index)

    def Active(self):
        m = self.Avg()
        e = self.Std()
        
        # change at least 2 times the standard errors
        return m.abs() > 2 * e

    def EffectsDiagnostics(self, cutoff = 0, yVar = 0, idx = 0):
        """ Plot a diagnostics by removing some factors using a
        cutoff for the effects and see you how well the prediction is.
        """

        table = pd.concat([self.table, self.interacts], axis=1)
        y = self.responses[idx].copy()
        eff = self.effects[idx].copy()
        cond = eff.iloc[:, yVar].abs() < cutoff
        eff[cond] = 0.0
        x = table.dot(eff/2) + y.mean()
        fig, ax = plt.subplots(1, 2, dpi=100, figsize=(5.25, 2.5))
        rmse = mean_squared_error(x, y, squared=False)
        ax[0].plot(y, x, 'r.', label=f"RMSE {rmse:0.2f}")
        xline = np.linspace(*ax[0].get_xlim())
        ax[0].plot(xline, xline, 'k:')
        ax[0].set(title = f"effect cutoff = {cutoff:.2f}",
               xlabel="True value", ylabel = "Prediction")
        ax[0].grid(True)
        ax[0].legend()

        ax[1].axhline(y=0, linestyle = "--", color='k')
        ax[1].plot(y, x - y, 'r.')
        ax[1].set(title = f"effect cutoff = {cutoff:.2f}",
               xlabel="True value", ylabel = "Residual")
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()
        return eff[~cond].iloc[:, yVar].sort_values(ascending=False)

    def PlotEffects(self, yVar=0):
        m = self.Avg()
        fig, ax = plt.subplots(1, 1, figsize=(3.25, 3.0))
        stats.probplot(m.iloc[:, yVar].values, dist="norm", plot=ax)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print(m.iloc[:, yVar].sort_values(ascending=False))
    
    def PlotResponse(self, yVar=0):
        a = pd.concat(self.responses)
        g = a.groupby(a.index)
        m = g.mean()
        fig, ax = plt.subplots(1, 1, figsize=(3.25, 3.0))
        stats.probplot(m.iloc[:, yVar].values, dist="norm", plot=ax)
        plt.tight_layout()

    def CubePlot(self, df, labels = None):
        """
            d.CubePlot(d.table.iloc[:,0:3], d.responses[0]['y'])
        """
        
        assert df.shape[1] >= 3, "At least 3 axes needed for a cube plot"
        if labels is not None:
            assert df.shape[0] == len(labels), "Labels and points size do not match"

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Get rid of colored axes planes
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        # Draw cube
        r = [-1,1]
        X, Y = np.meshgrid(r, r)
        one = np.ones(4).reshape(2, 2)
        ax.plot_wireframe(X,Y,one, color='k', alpha=0.5)
        ax.plot_wireframe(X,Y,-one, color='k', alpha=0.5)
        ax.plot_wireframe(X,-one,Y, color='k', alpha=0.5)
        ax.plot_wireframe(X,one,Y, color='k', alpha=0.5)
        ax.plot_wireframe(one,X,Y, color='k', alpha=0.5)
        ax.plot_wireframe(-one,X,Y, color='k', alpha=0.5)
        
        # Add points
        # ax.scatter3D(df.iloc[0:15, 0], df.iloc[0:15, 1], df.iloc[0:15, 2])
        
        if labels is not None:
            for r in range(df.shape[0]):
                ax.text(df.iloc[r, 0]-0.4, df.iloc[r, 1], df.iloc[r, 2]-0.2,
                        labels[r], fontsize=14, color='b')

        # Axis names
        ax.set_xlabel(df.columns[0], fontsize=14)
        ax.set_ylabel(df.columns[1], fontsize=14)
        ax.set_zlabel(df.columns[2], fontsize=14)
        ax.grid(False)
        
        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])
        ax.set_zticks([-1, 1])
        
        ax.view_init(-15, -30)
        plt.show()

    def Plot(self):
        self.PlotEffects()
        (
            self.Avg()
            .plot(kind='bar')
        )
        plt.show()
        return self.EffectsDiagnostics(0.1)

import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor

def predictfn(ml, df):
    yp = ml.Predict(df)
    ycol = ml.ytr.columns[0]
    # calculate effects in the original scale
    return ml.ScaleY(yp, False)[ycol].values[0]

def XGBBestFeatures(ml, design, xgparams, K = 5, cutoff = 0.08, save = False):
    print(f"Calculating XGB {K}-fold CV DOE values.")
    doeVals = pd.DataFrame(columns = list(range(K)), index = design.columns)

    for i in range(K):
        ml.SelectFold(ml.ytr.columns[0], design.factors.keys(), i, K)
        ml.Fit(xgb.XGBRegressor(**xgparams))
        design.run(ml, predictfn, trace=False)
        doeVals[i] = design.Avg()
        print("------------- K = %d OK" %(i+1))
        
    return process_feats("XGB", ml, K, cutoff, doeVals, save)


def GPRBestFeatures(ml, design, kernel, K = 5, cutoff = 0.08, save = False):
    print(f"Calculating GPR {K}-fold CV DOE values.")
    doeVals = pd.DataFrame(columns = list(range(K)), index = design.columns)

    for i in range(K):
        ml.SelectFold(ml.ytr.columns[0], design.factors.keys(), i, K)
        ml.Fit(GaussianProcessRegressor(kernel=kernel))
        design.run(ml, predictfn, trace=False)
        doeVals[i] = design.Avg()
        print("------------- K = %d OK" %(i+1))
        
    return process_feats("GPR", ml, K, cutoff, doeVals, save)

def process_feats(name, ml, K, cutoff, doeVals, save):
    if save:
        assert isinstance(save, str), "Please provide a file name"
    # Process and Plot Results
    doeVals['Avg. Effect'] = doeVals.mean(axis=1)
    doeVals['se'] = doeVals.std(axis=1)
    doeVals = doeVals.sort_values('Avg. Effect', ascending = True)

    fig, ax = plt.subplots(dpi=100, figsize=(5, doeVals.shape[0] * 0.18))
    doeVals[['Avg. Effect', 'se']].plot.barh(xerr = 'se', color='r', capsize=4, ax = ax)
    if cutoff > 0:
        ax.axvline(x=-cutoff, linestyle = "--", color = "b")
        ax.axvline(x=cutoff, linestyle = "--", color = "b")
    plt.title(f"DOE {K}-fold CV for y = {ml.ytr.columns[0]} ({name})")
    plt.xlabel("Effect +/- St. Dev.")
    if save:
        plt.savefig(save, dpi=600)
    else:
        plt.show()
    
    # Best Features
    cond = doeVals['Avg. Effect'].abs() < cutoff
    best = doeVals['Avg. Effect'].abs().sort_values(ascending = False)[~cond]
    print(best)

    bestset = []
    for feat in best.index:
        if "*" in feat:
            bestset += feat.split("*")
        else:
            bestset.append(feat)
            
    return sorted(list(set(bestset)))
