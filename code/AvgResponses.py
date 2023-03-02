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

plt.style.use("../../matplotlib.mplstyle")
mice = pd.read_csv("../../01_Imputation_MICE/imputed_data.mice.csv")

# Features to augment
xcols = ['lspk1', 'tspk1', 'lsfw1', 'tsfw1', 'lspk2', 'tspk2',
         'lsfw2', 'tsfw2', 'lspk3', 'tspk3', 'lsfw3', 'tsfw3']
ycols = ['square', 'full', 'lobe', 'half', 'agg', 'line']

# Gold
x1 = ['lspk1', 'tspk1', 'lsfw1', 'tsfw1', # peaks, fwhm
      'dp11', 'dw11', # shift, broadening
      # 'idp11', 'idw11' # inverse changes
     ]

# Coating
x2 = ['teos', 'lspk2','tspk2', 'lsfw2', 'tsfw2', # peaks, fwhm
      'lp21', 'tp21', 'lw21', 'tw21', 'dp22', 'dp21', 'dw22', 'dw21', # shift, broadening
      # 'idp22', 'idw22', 'idp21', 'idw21', 'ilp21', 'ilw21', 'itp21', 'itw21' # inverse changes
     ]

# Purification
x3 = ['lspk3', 'tspk3','lsfw3', 'tsfw3', # peaks, fwhm
      'lp31', 'lp32', 'tp31', 'tp32', 'lw31', 'lw32', 'tw31', 'tw32', 'dp33', 'dp31', 'dp32', 'dw33', 'dw31', 'dw32', # shift, broadening
      # 'idp33', 'idw33', 'idp31', 'idw31', 'ilp31', 'ilw31', 'itp31', 'itw31', 'idp32', 'idw32', 'ilp32', 'ilw32', 'itp32', 'itw32' # inverse changes
     ]

bestfeats3 = ['tspk1', 'tsfw3', 'lsfw2', 'lspk1', 'tspk2', 'lspk2', 'dw11', 'tp31', 'dp32']
bestfeats2 = ['tspk1', 'lspk1', 'lsfw2', 'tw21', 'dw11', 'tspk2', 'tsfw2', 'lspk2', 'tsfw1']

xgparams = {
    'max_depth': 2,
    'n_estimators': 1000,
    'eta': 0.02,
    'objective': 'reg:squarederror'
}

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
    df = mice[mice['.imp'] == imp]
    for opt in ("full", "lobe"):
        
        # XGB Regression Models
        # =============================================================
        xgmdl1 = BuildXGBModel("lobe", xgb.XGBRegressor(**xgparams), df)
        xgmdl2 = BuildXGBModel("full", xgb.XGBRegressor(**xgparams), df)
        xgmdl3 = BuildXGBModel("half", xgb.XGBRegressor(**xgparams), df)

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
        gpmdl3 = BuildGPRModel("half", GaussianProcessRegressor(kernel=kern), df)

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

