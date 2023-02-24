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
ycols = ['lobefrac', 'fullfrac']


#%% Prepare for training with the imputated dataset 1
reload(regressor)

ml = regressor.Regressor(dfi)
ml.AddFeatures(features.Differences, features.InverseDifferences)
ml.SetColumns(xcols, 'lobefrac')

# %% Gaussian Process Regression
kern = kernels.RBF() + kernels.WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kern)
ml.FitModel(gpr)

# %% Fitness
ml.Fitness(ml.PrepPrediction(df1))
ml.Fitness(ml.PrepPrediction(df2))
ml.Fitness(ml.PrepPrediction(df3))
ml.Fitness(ml.PrepPrediction(df4))
ml.Fitness(ml.PrepPrediction(df5))

# %% Random Forest Regressor
ml2 = regressor.Regressor(dfi)
ml2.AddFeatures(features.Differences, features.InverseDifferences)
ml2.SetColumns(xcols, 'lobefrac')

rfr = RandomForestRegressor(n_estimators=100, random_state=42)

ml2.FitModel(rfr)

# %% Fitness
ml2.Fitness(ml2.PrepPrediction(df1))
ml2.Fitness(ml2.PrepPrediction(df2))
ml2.Fitness(ml2.PrepPrediction(df3))
ml2.Fitness(ml2.PrepPrediction(df4))
ml2.Fitness(ml2.PrepPrediction(df5))

# %%
