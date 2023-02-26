#%%
import pandas as pd
from importlib import reload

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

from model import pipeline, regressor, features

#%% Load data
mice = pd.read_csv("Data/imputed_data.mice.csv")
compObs = mice[mice["imp"] == 0].dropna()

# Specify the features and target
xcols = ['lspk1', 'tspk1', 'lsfw1', 'tsfw1', 'lspk2', 'tspk2',
         'lsfw2', 'tsfw2', 'lspk3', 'tspk3', 'lsfw3', 'tsfw3']
ycol = "lobe"
# fnAgg = [features.Differences]
fnAgg = []

# %% Initialize
reload(pipeline)
reload(regressor)
models = []

# %% Linear Regression
lr = LinearRegression()

models.append(regressor.New(compObs, xcols, ycol, fnAgg))
models[-1].FitModel(lr)

# %% Gaussian Process Regression
kern = kernels.RBF(length_scale=[1.0] * len(xcols)) + kernels.WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kern)

models.append(regressor.New(compObs, xcols, ycol, fnAgg))
models[-1].FitModel(gpr)

# %% Random Forest Regression
rfr = RandomForestRegressor(n_estimators=1000, random_state=42)

models.append(regressor.New(compObs, xcols, ycol, fnAgg))
models[-1].FitModel(rfr)

# %% Perform test
for ml in models:
    res = ml.ParityAndResidual()
    print(res)

# %%
