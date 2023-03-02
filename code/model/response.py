from cProfile import label
import os
from itertools import product
from unittest import skip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import alphanums

import genetic

# cmap = "bwr"
# cmap = "cool"
cmap = "Spectral"

os.makedirs("response_surface", exist_ok=True)

def prepare_df(ml, default, col1, col2, unscale=False):
    s = 100
    
    assert isinstance(default, dict)
    assert np.all([c in default.keys() for c in ml.xCols])
    assert np.all([col1 in ml.xCols, col2 in ml.xCols]), \
        f"{col1} or {col2} not in training set. Make sure you're using the correct features!"

    # assuming a standard normal range
    x1 = np.linspace(-3, 3, num=s)
    x2 = np.linspace(-3, 3, num=s)

    arr = pd.DataFrame(list(product(x1, x2)),
                       columns = [col1, col2])

    for c in ml.xCols:
        if c not in [col1, col2]:
            arr[c] = default[c]
            
    z = ml.Predict(arr, scaleX=False, unscaleY=False)
    
    if unscale:
        arr = ml.UnscaleX(arr)
        z = ml.UnscaleY(z)
    
    x = arr[col1].values.reshape(s, s)
    y = arr[col2].values.reshape(s, s)
    z = z.values.reshape(s, s)
    
    # calc the optimum response
    Xdf = pd.DataFrame(default, columns = ml.xCols, index = [0])
    z0 = ml.Predict(Xdf, scaleX=False, unscaleY=False)
    x0 = default[col1]
    y0 = default[col2]
    
    if unscale:
        Xdf = ml.UnscaleX(Xdf)
        z0 = ml.UnscaleY(z0)
        x0 = Xdf[col1]
        y0 = Xdf[col2]
    
    return (x, y, z), (x0, y0, z0)

def Plot3d(ml, default, col1, col2,
           unscale=True, invertcmap=False, save = False):
    (x, y, z), (x0, y0, z0) = prepare_df(ml, default, col1, col2, unscale)
    
    oldRc = plt.rcParams
    plt.rcParams['grid.color'] = "#00000033"
    plt.rcParams['xtick.minor.visible'] = False
    plt.rcParams['ytick.minor.visible'] = False
    
    colmap = cmap+"_r" if invertcmap else cmap

    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, 
                           alpha=0.7,
                           cmap=colmap,
                           linewidth=0,
                           antialiased=True)
    ax.scatter(x0, y0, z0, marker = '*', color = 'k', s=40)
    ax.set_xlabel(col1, color = 'b')
    ax.set_ylabel(col2, color = 'b')
    ax.set_zlabel(ml.yCol, color='r')

    if save:
        plt.savefig(f"response_surface/{save}", dpi=300)
        plt.close(fig)
    else:
        plt.show()
    plt.rcParams = oldRc

def PlotContour(ml, default, col1, col2,
                unscale=True, invertcmap=False, save = False):
    (x, y, z), (x0, y0, z0) = prepare_df(ml, default, col1, col2, unscale)
    
    colmap = cmap+"_r" if invertcmap else cmap
    z = np.round(z, 2)

    fig, ax = plt.subplots(figsize=(3.25, 3.0), dpi=100)
    CS = ax.contour(x, y, z, cmap=colmap, linewidths=1)
    ax.scatter(x0, y0, s=40, marker = '*', color = 'k')
    ax.clabel(CS, inline=True, fontsize=9, colors='k')
    ax.set_xlabel(col1, color = 'b')
    ax.set_ylabel(col2, color = 'b')
    ax.set_title(f"Prediction for {ml.yCol}")
    ax.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(f"response_surface/{save}", dpi=300)
        plt.close(fig)
    else:
        plt.show()

# If the response is normalized [0, 1]
def MinimizeNorm(y, k=6):
    return 1.0 / (np.exp(k * y))

def MaximizeNorm(y, k=12):
    return 1.0 / (1 + np.exp(k / 2 - k * y))

def GAOptimize(x0, *models, **kwargs):
    assert isinstance(x0, pd.Series), \
        "Please provide scaled x0 as a Series"
    for spec in models:
        assert len(spec) > 1, \
            "Please provide (model, transformation fn, [k])"
        assert callable(spec[1]), \
            "Please provide (model, transformation fn, [k])"

    m = len(models)
    fvals = []
    
    def objective(X):
        prod = np.ones(X.shape[0])
        Xdf = pd.DataFrame(X, columns=x0.index)

        k = None
        yvals = []
        dvals = []

        for spec in models:
            try:
                ml, fn, k = spec
            except:
                ml, fn = spec

            y = ml.Predict(Xdf, scaleX=False).values
            if k is None:
                d = fn(y)
            else:
                d = fn(y, k)
                
            d[d > 1] = 1.0
            d[d < 0] = 0.0
            prod = prod * d

            # max or min?
            v = -1 if fn(10) > 0.5 else 0
            yvals.append(np.sort(y)[v])
            dvals.append(np.sort(d)[-1])
                
        D = prod**(1/m)
        fvals.append((np.sort(D)[-1], yvals, dvals))
        # Maximize
        return -1.0 * D

    res = genetic.Minimize(objective, x0.values, **kwargs)
    obj = (
        dict(zip(x0.index, res)),
        [(ml[0], ml[1](10) > 0.5) for ml in models],
        fvals,
    )
    return obj

def PlotDesired(result, save=False):
    assert len(result) >= 3
    if save:
        assert isinstance(save, str), \
            "Please provide a file name."

    fig, ax = plt.subplots(3, 1, figsize=(5, 5), sharex=True)
    yCols = [ml[0].yCol for ml in result[1]]
    fvals = [y[0] for y in result[2]]
    yvals = [y[1] for y in result[2]]
    dvals = [y[2] for y in result[2]]

    ax[0].plot(fvals, label='- D')
    for i in range(len(yvals[0])):
        ax[2].plot([y[i] for y in yvals], "-", label=yCols[i])
        ax[1].plot([d[i] for d in dvals], "-", label=yCols[i])

    ax[0].set(ylabel ="best f-value")
    ax[1].set(ylabel ="best d-value")
    ax[2].set(ylabel ="best y-value", xlabel="Iteration")

    for a in ax.flatten():
        a.legend()
        a.grid()

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=600)
        plt.close(fig)
        print("Saved:", save)
    else:
        plt.show()

def PlotDesiredDistance(result, df0, N=3, save=False):
    assert len(result) >= 2
    assert isinstance(N, int)
    assert isinstance(df0, pd.DataFrame), f"df0 must be a DataFrame, not {type(df0)}"
    if save:
        assert isinstance(save, str), "Please provide a file name."

    xMax = result[0]
    ml = result[1][0][0]


    df = ml.Prep(df0)
    for c in ml.xCols:
        assert c in df.columns, f"{c} for {ml} not in df0"

    dX = df[ml.xCols]
    opt = ml.UnscaleX(ml.Prep(xMax))
    dist = np.linalg.norm(dX.values - opt.values, axis=1)
    cl = np.argsort(dist)

    fig, ax = plt.subplots()
    ax.plot(dX.index, dist)
    ax.set(xlabel="Index", ylabel="Eucledian distance")
    ax.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=600)
        plt.close(fig)
    else:
        plt.show()

    df = df.assign(distance = dist)
    return pd.concat([opt.T, df.iloc[cl[0:N]].T], axis=1)

def PlotDesiredPairs(result, cols = None, save = False, surface = True, contour = True):
    assert len(result) >= 2
    assert isinstance(save, bool)
    xMax = result[0]
    models = result[1]

    for ml, maximize in models:
        if cols is None:
            xcols = ml.xCols
        else:
            xcols = cols
        print(ml)

        for c in range(len(xcols)):
            col1 = xcols[c]
            for d in range(c+1, len(xcols)):
                col2 = xcols[d]
                print(f"Plotting {ml.yCol} = f({col1}, {col2})")
                savename = f"{col1}_{col2}.{ml.yCol}_{ml.Name}.png"
                if surface:
                    fname = "desired_surface." + savename if save else False
                    Plot3d(ml, xMax, col1, col2, invertcmap = maximize,
                           save = fname)
                if contour:
                    fname = "desired_contour."+savename if save else False
                    PlotContour(ml, xMax, col1, col2, invertcmap = maximize,
                                save = fname)

def CalcRobustness(result, df, cols = None, othercenter=None, otherscale=0.5, save = False):
    assert len(result) >= 2
    assert isinstance(save, bool)
    assert isinstance(df, pd.DataFrame), f"df must be a DataFrame, not {type(df)}"

    if othercenter is None:
        print(f"Calculating uncertainty with other features N(loc=optimum, {otherscale})")
    else:
        print(f"Calculating uncertainty with other features N(loc={othercenter}, {otherscale})")

    sMax = result[0]
    ml = result[1][0][0]
    if cols is None:
        cols = sMax.keys()

    uMax = ml.UnscaleX(ml.Prep(sMax))
    df = ml.Prep(df)

    colors = ["#" + c for c in ['0C5DA5', 'FF2C00', '00B945', 'FF9500', '845B97', '474747', '9e9e9e']]

    for col in cols:
        fig, ax = plt.subplots(2, 1, figsize=(3.25, 4.2), dpi=100, sharex=True)
        opt = uMax[col].values[0]
        x = opt
        s = 20

        # Plot each of the response predictions
        for j, (ml, maximize) in enumerate(result[1]):
            # Make exact predictions with the optimum
            xpred = {}
            for c in sMax.keys():
                xpred[c] = s * [sMax[c]]

            xpred[col] = np.linspace(-6, 6, num=s)
            xpred = pd.DataFrame(xpred)
            ypred = ml.Predict(xpred, scaleX = False)
            uxpred = ml.UnscaleX(xpred)

            # Make uncertainty predictions around the optimum
            vpred = ypred.copy()
            for i in range(10):
                xpred = {}
                for c in sMax.keys():
                    xpred[c] = np.random.normal(loc=sMax[c] if othercenter is None else othercenter,
                                scale=otherscale)

                xpred[col] = np.linspace(-6, 6, num=s)
                xpred = pd.DataFrame(xpred)
                ypred = ml.Predict(xpred, scaleX = False)
                vpred = pd.concat([vpred, ypred], axis=1)

            vpred = vpred.std(axis=1)
            ax[0].errorbar(uxpred[col], ypred, yerr=vpred, color=colors[j],
                            alpha=0.6, fmt="None", capsize=2)

            ax[0].plot(uxpred[col], ypred, '.-', color=colors[j], label=ml.yCol)
            ax[0].set(ylabel="Prediction +/- St. Dev.")

        histy, bins, patches = ax[1].hist(df[col].values, density=False)
        y = np.mean(histy)
        for i in range(len(bins)-1):
            if opt > bins[i] and opt < bins[i+1]:
                y = histy[i]
                x = bins[i] + 0.5 * (bins[i+1] - bins[i])
                break

        ax[1].scatter(x, y, marker='*', c='red', s=40, alpha=1, label=f"{opt:0.2f}")
        ax[1].set(ylabel=f"Frequency", xlabel=col)

        ax[0].legend()
        ax[1].legend()

        binwidth = bins[1] - bins[0]
        ax[1].set_xlim(min(opt, bins[0]) - binwidth, max(opt, bins[-1]) + binwidth)

        plt.tight_layout()        

        if save:
            fname = f"response_surface/robustness1d.{col}_{ml.Name}.png"
            plt.savefig(fname, dpi=300)
            plt.close(fig)
        else:
            plt.show()
    