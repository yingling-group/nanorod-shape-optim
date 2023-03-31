#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt

import regressor
import features
import response

import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

plt.style.use("../../matplotlib.mplstyle")
mice = pd.read_csv("../../01_Imputation_MICE/imputed_data.mice.csv")

# Features to augment
xcols = ['lspk1', 'tspk1', 'lsfw1', 'tsfw1', 'lspk2', 'tspk2',
         'lsfw2', 'tsfw2', 'lspk3', 'tspk3', 'lsfw3', 'tsfw3']
ycols = ['square', 'full', 'lobe', 'half', 'agg', 'line']

bestfeats3 = ['tspk1', 'tsfw3', 'lsfw2', 'lspk1', 'tspk2', 'lspk2', 'dw11', 'tp31', 'dp32']
bestfeats2 = ['tspk1', 'lspk1', 'lsfw2', 'tw21', 'dw11', 'tspk2', 'tsfw2', 'lspk2', 'tsfw1']

def BuildModel(X, y, alg, df):
    ml = regressor.Regressor(df)
    ml.Augment(xcols+ycols, K=3, scale=0.2, plot=False)
    ml.AddFeatures(features.Differences, show_list=False)
    ml.Fit(X, y, alg, plot=False)
    return ml

xbest = bestfeats3
df = mice[mice['.imp'] == 3]

# -----------------------------------------------------------------------
xgparams = {
    'max_depth': 2,
    'n_estimators': 1000,
    'eta': 0.02,
    'objective': 'reg:squarederror'
}

xgmdl1 = BuildModel(xbest, "lobe", xgb.XGBRegressor(**xgparams), df)
xgmdl2 = BuildModel(xbest, "full", xgb.XGBRegressor(**xgparams), df)
xgmdl3 = BuildModel(xbest, "half", xgb.XGBRegressor(**xgparams), df)
# xgmdl4 = BuildModel(xbest, "agg", xgb.XGBRegressor(**xgparams), df)

x0 = xgmdl1.ScaleX(xgmdl1.Prep(df)).mean()
res = response.GAOptimize(x0, 
                [xgmdl2, response.MaximizeNorm],
                [xgmdl3, response.MinimizeNorm],
                # [xgmdl4, response.MinimizeNorm],
                [xgmdl1, response.MinimizeNorm]
)

response.PlotDesired(res, "optimization_xgb.png")
dist = response.PlotDesiredDistance(res, mice[mice[".imp"] == 3], save="optimum_distance_xgb.png")
print(dist)
response.CalcRobustness(res, mice[mice[".imp"] == 3], otherscale=0.5, save=True)
# response.PlotDesiredPairs(res, xbest, save=True)

# -----------------------------------------------------------------------

kern = kernels.RBF([1] * len(xbest)) + kernels.WhiteKernel()

gpmdl1 = BuildModel(xbest, "lobe", GaussianProcessRegressor(kernel=kern), df)
gpmdl2 = BuildModel(xbest, "full", GaussianProcessRegressor(kernel=kern), df)
gpmdl3 = BuildModel(xbest, "half", GaussianProcessRegressor(kernel=kern), df)
# gpmdl4 = BuildModel(xbest, "agg", GaussianProcessRegressor(kernel=kern), df)

x0 = gpmdl1.ScaleX(gpmdl1.Prep(df)).mean()
res = response.GAOptimize(x0, 
                [gpmdl2, response.MaximizeNorm],
                [gpmdl3, response.MinimizeNorm],
                # [gpmdl4, response.MinimizeNorm],
                [gpmdl1, response.MinimizeNorm]
)

response.PlotDesired(res, "optimization_gpr.png")
dist = response.PlotDesiredDistance(res, mice[mice[".imp"] == 3], save="optimum_distance_gpr.png")
print(dist)
response.CalcRobustness(res, mice[mice[".imp"] == 3], otherscale=0.5, save=True)
# response.PlotDesiredPairs(res, xbest, save=True)

print("Done!")

