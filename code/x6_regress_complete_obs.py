#%%
import os
import argparse
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

from model import regressor, features, utils

parser = argparse.ArgumentParser(
    description = 'Run regression on the saved dataset',
    epilog = 'Author: Akhlak Mahmood, Yingling Group, NCSU')

parser.add_argument('--show', help="Display the plots", action="store_true")
cargs = parser.parse_args()

os.makedirs("Plots", exist_ok=True)
output = "Data/K-fold_cv_summary_regression.csv"

results = utils.dfBuilder()

# %%
xgparams = {
    'max_depth': 20,
    'n_estimators': 100,
    'eta': 0.05,
    'objective': 'reg:squarederror'
}


# Define the models to run
def models(xlen):
    lr = LinearRegression()
    kern = kernels.RBF() + kernels.WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kern)
    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    xgbr = xgb.XGBRegressor(**xgparams)

    return [gpr, xgbr]

def run_kfold_cv(K, xlen, name, *args):
    """ Run K-fold on the given models and dataset """
    for fold in range(K):
        for reg in models(xlen):
            ml = regressor.New(*args)
            ml.Split(fold=fold, K=K)
            ml.FitModel(reg)
            res = ml.ParityAndResidual(show=cargs.show)
            results.add(name = name, model = ml.Name, K = K,
                        fold = fold, R2 = res['R2'], RMSE = np.sqrt(res['MSE']))

#%% Load data
mice = pd.read_csv("Data/imputed_data.mice.csv")

# Specify the target
ycol = "full"

# Find the non-collinear features
fnAgg = [features.Differences, features.InverseDifferences]
df = mice[mice.imp >= 0].dropna().drop(['imp', 'id'], axis=1).set_index('name')
reg = regressor.Regressor(df)
reg.AddFeatures(fnAgg, show_list=False)
feats = reg.NonCollinearFeatures(keepCols=['teosVolPct', 'teosVolume'],
                                 ignoreCols=['lobe', 'full', 'other', 'quality'])

feats.to_csv("Data/selected_features.csv")
xcols = list(feats.columns)

# %% Run

K = 6 # number of CV folds

compObs = mice[mice["imp"] == 0].dropna()

# Shuffle the data
compObs = compObs.sample(frac=1.0)

run_kfold_cv(K, len(xcols), "completeObs",
             compObs, xcols, ycol, fnAgg)

for i in range(1, 6):
    name = "imputedObs%d" %i
    imputedObs = mice[mice["imp"] == i]

    # Shuffle the data
    imputedObs = imputedObs.sample(frac=1.0)

    run_kfold_cv(K, len(xcols), name,
                imputedObs, xcols, ycol, fnAgg)


# %% Print and save the metrics
print(results)

summary = utils.summarize_results(results.df, ["model", "name", "K"],
                                  imputeCol="name", ignoreValues="completeObs", includeIndividual=False)
print(summary)

summary.to_csv(output, index=False)
print("Save OK:", output)
print(feats)