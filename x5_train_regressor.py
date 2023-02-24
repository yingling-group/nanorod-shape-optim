#%%
import os 
from importlib import reload

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

from model import regressor, features

#%% Training data
train = pd.read_csv("Data/imputed_data.mice.csv")
train.head()

dfi = train[train["imp"] != 0]
df1 = train[train["imp"] == 1]
df2 = train[train["imp"] == 2]
df3 = train[train["imp"] == 3]
df4 = train[train["imp"] == 4]
df5 = train[train["imp"] == 5]

# Specify the features and target
xcols = ['lspk1', 'tspk1', 'lsfw1', 'tsfw1', 'lspk2', 'tspk2',
         'lsfw2', 'tsfw2', 'lspk3', 'tspk3', 'lsfw3', 'tsfw3']
ycol = "lobefrac"
fnAgg = [features.Differences, features.InverseDifferences]


# %% Gaussian Process Regression
reload(regressor)

ml1 = regressor.New(dfi, xcols, ycol, fnAgg)
kern = kernels.RBF() + kernels.WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kern)
ml1.FitModel(gpr)

# %% Random Forest Regressor
ml2 = regressor.New(dfi, xcols, ycol, fnAgg)
rfr = RandomForestRegressor(n_estimators=1000, random_state=42)
ml2.FitModel(rfr)

# %% Load the testing data
testdf = pd.read_csv("Data/testing_spectra.csv")

ml1.ParityAndResidual(testdf)
ml2.ParityAndResidual(testdf)

# %%
