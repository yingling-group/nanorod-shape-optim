import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

from model import regressor, features, utils

os.makedirs("Plots", exist_ok=True)

inpMice = "Data/imputed_data.mice.csv"
outResults = "Data/x7_ensemble_summary.csv"
outFeats = "Data/x7_selected_features.csv"

plt.style.use("matplotlib.mplstyle")

#%% INITIALIZE
# =============================================================================
parser = argparse.ArgumentParser(
    description = 'Run regression on the imputed datasets (v2023.02)',
    epilog = 'Author: Akhlak Mahmood, Yingling Group, NCSU')

parser.add_argument('--show', help="Display the plots", action="store_true")
print("\n", parser.description, "\n\n", parser.epilog, "\n")

cargs = parser.parse_args()
results = utils.dfBuilder()

#%% SETUP DATA
# =============================================================================
mice = pd.read_csv(inpMice)

# Sample 6 data points for testing
testIds = mice[mice.imp == 0].dropna().id.sample(n=6)
print("Test Row IDs:", testIds.values)

# Separate test and training data
testdf = mice[mice.imp == 0].loc[lambda df: df.id.isin(testIds)]
traindf = mice[mice.imp > 0].loc[lambda df: ~df.id.isin(testIds)]

# List of imputed dataframes
impdfs = [traindf[traindf.imp == i] for i in traindf.imp.unique()]

#%% SETUP FEATURES
# =============================================================================
ycol = "lobe"

# Build a list of non-collinear features
fnAgg = [features.Differences, features.InverseDifferences]
df = mice[mice.imp >= 0].dropna().drop(['imp', 'id'], axis=1).set_index('name')
reg = regressor.Regressor(df)
reg.AddFeatures(fnAgg, show_list=False)
feats = reg.NonCollinearFeatures(keepCols=['teosVolPct', 'teosVolume'],
                                 ignoreCols=['lobe', 'full', 'other', 'quality'])

feats.to_csv(outFeats)
xcols = list(feats.columns)

print("Selected features:", xcols)

#%% SETUP ALGORITHMS
# =============================================================================

kern = kernels.RBF() #+ kernels.WhiteKernel()
xgparams = {
    'max_depth': 20,
    'n_estimators': 100,
    'eta': 0.05,
    'objective': 'reg:squarederror'
}

algorithms = {
    "LR": (LinearRegression, {}),
    "GPR": (GaussianProcessRegressor, dict(kernel=kern)),
    "RF": (RandomForestRegressor, dict(n_estimators=100, random_state=42)),
    "XGBR": (xgb.XGBRegressor, xgparams),
}

#%% RUN TRAINING AND TESTING
# =============================================================================
# Evaluate each algorithms separately
for name in algorithms:
    # Train a set of models on each imputed dataset for each algorithm
    enreg = regressor.EnsembleRegressor(name, xcols, ycol, fnAgg)
    enreg.TrainOnData(impdfs, algorithms[name])
    res = enreg.ParityScore(testdf, savePrefix = "Plots/ensembleAvg",
                            show=cargs.show)

    results.add(model = name,
                R2 = res['R2'], RMSE = np.sqrt(res['MSE']))


#%% FINISH UP
# =============================================================================
print()
print(results)
results.df.to_csv(outResults, index=False)

print("Save OK:", outResults)
print("Save OK:", outFeats)
