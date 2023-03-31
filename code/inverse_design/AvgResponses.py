#!/usr/bin/env python
import os
from re import A
import pandas as pd
import matplotlib.pyplot as plt

import regressor
import features
import response

import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

plt.style.use("../matplotlib.mplstyle")
mice = pd.read_csv("../../Data/imputed_data.mice.csv")

# Features to augment
xcols = ['lspk1', 'tspk1', 'lsfw1', 'tsfw1', 'lspk2', 'tspk2',
         'lsfw2', 'tsfw2', 'lspk3', 'tspk3', 'lsfw3', 'tsfw3']
ycols = ['full', 'other', 'lobe']

# Gold
x1 = ['lspk1', 'tspk1', 'lsfw1', 'tsfw1', # peaks, fwhm
      'dp11', 'dw11', # shift, broadening
     ]

# Coating
x2 = ['teos', 'lspk2','tspk2', 'lsfw2', 'tsfw2', # peaks, fwhm
      'lp21', 'tp21', 'lw21', 'tw21', 'dp22', 'dp21', 'dw22', 'dw21', # shift, broadening
     ]

# Purification
x3 = ['lspk3', 'tspk3','lsfw3', 'tsfw3', # peaks, fwhm
      'lp31', 'lp32', 'tp31', 'tp32', 'lw31', 'lw32', 'tw31', 'tw32', 'dp33', 'dp31', 'dp32', 'dw33', 'dw31', 'dw32', # shift, broadening
     ]

bestfeats3 = ['tspk1', 'tsfw3', 'lsfw2', 'lspk1', 'tspk2', 'lspk2', 'dw11', 'tp31', 'dp32']
bestfeats2 = ['tspk1', 'lspk1', 'lsfw2', 'tw21', 'dw11', 'tspk2', 'tsfw2', 'lspk2', 'tsfw1']

xgparams = dict(base_score=0.5, booster='gbtree', callbacks=None,
    colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
    early_stopping_rounds=None, enable_categorical=False,
    eval_metric=None, gamma=0.0, gpu_id=-1, grow_policy='depthwise',
    importance_type=None, interaction_constraints='',
    learning_rate=0.1, max_bin=256, max_cat_to_onehot=4,
    max_delta_step=0, max_depth=5, max_leaves=0, min_child_weight=1,
    monotone_constraints='()', n_estimators=8, n_jobs=0,
    num_parallel_tree=1, objective='reg:squarederror', predictor='auto',
    random_state=0, reg_alpha=0.4)

def BuildXGBModel(y, alg, df):
    ml = regressor.Regressor(df)
    ml.Augment(xcols+ycols, K=4, scale=0.10, plot=False)
    ml.AddFeatures(features.Differences, show_list=False)
    ml.Fit(bestfeats3, y, alg, plot=False)
    return ml

kern = kernels.RBF() + kernels.WhiteKernel()

def BuildGPRModel(y, alg, df):
    ml = regressor.Regressor(df)
    ml.Augment(xcols+ycols, K=3, scale=0.10, plot=False)
    ml.AddFeatures(features.Differences, show_list=False)
    ml.Fit(bestfeats3, y, alg, plot=False)
    return ml

allres = {
    "imp": [],
    "alg": [],
    "opt": []
}

for imp in range(1, 6):
    df = mice[mice['imp'] == imp]
    for opt in ("full", "lobe"):
        
        # XGB Regression Models
        # =============================================================
        xgmdl1 = BuildXGBModel("lobe", xgb.XGBRegressor(**xgparams), df)
        xgmdl2 = BuildXGBModel("full", xgb.XGBRegressor(**xgparams), df)
        xgmdl3 = BuildXGBModel("other", xgb.XGBRegressor(**xgparams), df)

        x0 = xgmdl1.ScaleX(xgmdl1.Prep(df)).mean()
        res = response.GAOptimize(x0, 
                        [xgmdl1, response.MaximizeNorm if opt == "lobe" else response.MinimizeNorm],
                        [xgmdl2, response.MaximizeNorm if opt == "full" else response.MinimizeNorm],
                        [xgmdl3, response.MinimizeNorm]
        )
        res = xgmdl1.UnscaleX(xgmdl1.Prep(res[0]))
        for col in res:
            if col not in allres:
                allres[col] = []
            allres[col].append(res[col].values[0])
        allres['imp'].append(imp)
        allres['alg'].append('xgb')
        allres['opt'].append(opt)
        print(res)

        # GPR Regression Models
        # =============================================================
        gpmdl1 = BuildGPRModel("lobe", GaussianProcessRegressor(kernel=kern), df)
        gpmdl2 = BuildGPRModel("full", GaussianProcessRegressor(kernel=kern), df)
        gpmdl3 = BuildGPRModel("other", GaussianProcessRegressor(kernel=kern), df)

        x0 = gpmdl1.ScaleX(gpmdl1.Prep(df)).mean()
        res = response.GAOptimize(x0, 
                        [gpmdl1, response.MaximizeNorm if opt == "lobe" else response.MinimizeNorm],
                        [gpmdl2, response.MaximizeNorm if opt == "full" else response.MinimizeNorm],
                        [gpmdl3, response.MinimizeNorm]
        )

        res = xgmdl1.UnscaleX(xgmdl1.Prep(res[0]))
        for col in res:
            if col not in allres:
                allres[col] = []
            allres[col].append(res[col].values[0])
        allres['imp'].append(imp)
        allres['alg'].append('gpr')
        allres['opt'].append(opt)
        print(res)

        print(f"Imp{imp}-{opt} OK\n")

df = pd.DataFrame(allres)
print(df)

df.to_csv("AvgResponses.csv")
print("Done!")

