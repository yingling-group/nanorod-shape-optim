#%%
import pandas as pd
from importlib import reload

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

from model import regressor, features

#%% Training data
train = pd.read_csv("Data/imputed_data.mice.csv")
dfi = train[train["imp"] != 0]

# Specify the features and target
xcols = ['lspk1', 'tspk1', 'lsfw1', 'tsfw1', 'lspk2', 'tspk2',
         'lsfw2', 'tsfw2', 'lspk3', 'tspk3', 'lsfw3', 'tsfw3']
ycol = "lobefrac"
fnAgg = [features.Differences]


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

res1 = ml1.ParityAndResidual(testdf)
res2 = ml2.ParityAndResidual(testdf)

print(res1)
print(res2)

# %%
