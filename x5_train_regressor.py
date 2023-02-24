#%%
import os 
from importlib import reload

import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

from model import regressor, features

#%% Training data
train = pd.read_csv("Data/imputed_data.mice.csv")
train.head()

# Specify the features and target
xcols = ['lspk1', 'tspk1', 'lsfw1', 'tsfw1', 'lspk2', 'tspk2',
         'lsfw2', 'tsfw2', 'lspk3', 'tspk3', 'lsfw3', 'tsfw3']
ycols = ['lobefrac', 'fullfrac']


#%% Prepare for training with the imputated dataset 1
reload(regressor)

df1 = train[train["imp"] == 1]
df2 = train[train["imp"] == 2]

ml = regressor.Regressor(df1)
ml.AddFeatures(features.Differences)
ml.SetColumns(xcols, 'fullfrac')

# %% Gaussian Process Regression
kern = kernels.RBF() + kernels.WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kern)
ml.FitModel(gpr)

# %% Fitness
ml.Fitness(ml.PrepPrediction(df2))
