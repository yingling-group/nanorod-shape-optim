#%%
import os
import argparse
import numpy as np
import pandas as pd

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
# Define the models to run
def models(xlen):
    lr = LinearRegression()
    kern = kernels.RBF(length_scale=[1.0] * xlen) + kernels.WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kern)
    rfr = RandomForestRegressor(n_estimators=1000, random_state=42)

    return [lr, gpr, rfr]

def run_kfold_cv(K, xlen, name, *args):
    """ Run K-fold on the given models and dataset """
    for fold in range(K):
        for reg in models(xlen):
            ml = regressor.New(*args)
            ml.Split(fold=fold, K=K)
            ml.FitModel(reg)
            res = ml.ParityAndResidual(show=cargs.show)
            results.add(name = name, model = ml.Name, K = K,
                        fold = fold, r2 = res['R2'], rmse = np.sqrt(res['MSE']))

#%% Load data
mice = pd.read_csv("Data/imputed_data.mice.csv")

# Specify the features and target
ycol = "lobe"
xcols = ['lspk1', 'tspk1', 'lsfw1', 'tsfw1', 'lspk2', 'tspk2',
         'lsfw2', 'tsfw2', 'lspk3', 'tspk3', 'lsfw3', 'tsfw3']

# Feature aggregation function
fnAgg = []
fnAgg = [features.Differences]

# %%
# Select target data
compObs = mice[mice["imp"] == 0].dropna()

# %% Run
run_kfold_cv(5, len(xcols), "completeObs",
             compObs, xcols, ycol, fnAgg)


# %% Print and save the metrics
print(results)

summary = results.df.groupby(['name', 'model']).agg([np.mean, np.std])
print(summary)

summary.to_csv(output, index=True)
print("Save OK:", output)
