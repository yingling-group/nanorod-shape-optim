#%%
import os 
from importlib import reload
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

import regressor
import features
import genetic
import response

#%%
plt.style.use("../../matplotlib.mplstyle")
mice = pd.read_csv("../../01_Imputation_MICE/imputed_data.mice.csv")

# Features to augment
xcols = ['lspk1', 'tspk1', 'lsfw1', 'tsfw1', 'lspk2', 'tspk2',
         'lsfw2', 'tsfw2', 'lspk3', 'tspk3', 'lsfw3', 'tsfw3']
ycols = ['square', 'full', 'lobe', 'half', 'agg', 'line']

bestfeats3 = ['tspk1', 'tsfw3', 'lsfw2', 'lspk1', 'tspk2', 'lspk2', 'dw11', 'tp31', 'dp32']
bestfeats2 = ['tspk1', 'lspk1', 'lsfw2', 'tw21', 'dw11', 'tspk2', 'tsfw2', 'lspk2', 'tsfw1']

#%%
df0 = mice[mice[".imp"] == 1]
xbest = bestfeats3

xgparams = {
    'max_depth': 2,
    'n_estimators': 1000,
    'eta': 0.05,
    'objective': 'reg:squarederror'
}

kern = kernels.RBF() + kernels.WhiteKernel()

#%%

reload(regressor)
ml = regressor.Regressor(df0)
ml._set_cols(xcols, 'lobe')
ml.ScaleX()
ml.UnscaleX()
ml.ScaleY(df0['lobe'])
ml.UnscaleY()
ml.Augment(xcols+ycols, df0, scale=0.2)
# ml.Compare(df0[xcols], scaled=True)
ml.AddFeatures(features.Differences)
# ml.Split(xbest, 'lobe')
ml.Split(xbest, 'lobe', fold=0, K=5)


#%%
reload(regressor)

ml = regressor.Regressor(df0)
ml.Augment(xcols+ycols, K=3, scale=0.2, plot=False)
# ml.Compare(df[xcols], scaled=False)
df = ml.AddFeatures(features.Differences, show_list=False)

for i in range(5):
    ml.Split(xbest, 'lobe', df=df, fold=i, K=5)
    ml.FitModel(xgb.XGBRegressor(**xgparams))
# %%
for j in range(1, 6):
    ml.Accuracy(mice[mice[".imp"] == j])
# %%
reload(regressor)

ml = regressor.Regressor(df0)
ml.Augment(xcols+ycols, K=3, scale=0.2, plot=False)
# ml.Compare(df[xcols], scaled=False)
df = ml.AddFeatures(features.Differences, show_list=False)

for i in range(5):
    ml.Split(xbest, 'lobe', df=df, fold=i, K=5)
    ml.FitModel(GaussianProcessRegressor(kernel=kern))

# %%
reload(genetic)
reload(response)
ml = regressor.Regressor(df0)
ml.Augment(xcols+ycols, K=3, scale=0.2, plot=False)
ml.AddFeatures(features.Differences, show_list=False)
ml.Fit(xbest, 'lobe', GaussianProcessRegressor(kern))

x0 = ml.ScaleX(ml.Prep(mice[mice['.imp'] >= 1])).mean()

res = response.GAOptimize(x0,
                [ml, response.MaximizeNorm]
)

response.PlotDesired(res)
response.PlotDesiredDistance(res, mice[mice[".imp"] >= 1])
response.PlotDesiredPairs(res, xbest)
# %%
reload(response)
response.CalcRobustness(res, mice[mice[".imp"] == 1], othercenter=None, otherscale=1)
# %%
