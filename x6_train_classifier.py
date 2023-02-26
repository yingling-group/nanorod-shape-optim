#%%
import pandas as pd
from importlib import reload

from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import kernels

from model import pipeline, classifier, features

#%% Training data
train = pd.read_csv("Data/imputed_data.mice.csv")
dfi = train[train["imp"] != 0]
dfi = dfi.assign(nrshape = "full")
dfi.loc[dfi.lobefrac > 0.5, "nrshape"] = "lobe"

# Specify the features and target
xcols = ['lspk1', 'tspk1', 'lsfw1', 'tsfw1', 'lspk2', 'tspk2',
         'lsfw2', 'tsfw2', 'lspk3', 'tspk3', 'lsfw3', 'tsfw3']
ycol = "nrshape"
fnAgg = [features.Differences]

# %% Gaussian Process Regression
reload(pipeline)
reload(classifier)

ml1 = classifier.New(dfi, xcols, ycol, fnAgg)
kern = kernels.RBF(length_scale=[1.0] * len(xcols)) + kernels.WhiteKernel()
gpc = GaussianProcessClassifier(kernel=kern)
ml1.FitModel(gpc)

# %% Random Forest Regression
ml2 = classifier.New(dfi, xcols, ycol, fnAgg)
rfr = RandomForestClassifier(n_estimators=1000, random_state=42)
ml2.FitModel(rfr)

# %% Load the testing data
testdf = pd.read_csv("Data/testing_spectra.csv")

res1 = ml1.ParityAndResidual(testdf)
res2 = ml2.ParityAndResidual(testdf)

print(res1)
print(res2)

# %%
