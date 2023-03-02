#!/usr/bin/env python
# coding: utf-8

# ```bash
# ## TO RUN THIS NOTEBOOK FROM THE TERMINAL
# $ jupyter nbconvert --to script x10_classifier_pipeline.ipynb
# $ python x10_classifier_pipeline.py
# ```

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import BorderlineSMOTE

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import kernels, GaussianProcessClassifier


# ### Load libraries

# In[ ]:


# Custom scikit-learn like pipeline with additional functionalities
import pipeline as pl


# In[ ]:


# How the pipeline should be run for this project
from model.AdData import *
from model.AdFeatures import *
from model.AdClassify import TestPerformance


# In[ ]:


# Utilities
from model import hyperparams
from model import plotlib


# ### Initialize

# In[ ]:


plotlib.load_fonts("../../../common/fonts/")
plt.style.use("matplotlib.mplstyle")

inputCsv = "../Data/imputed_data.mice.csv"
ignoreXCols = ['imp', 'id', 'quality', 'lobe', 'full', 'other', 'coatingId']


# In[ ]:


loader = LoadData()
loader.Execute(inputCsv)


# In[ ]:


plotQuality = pl.PlotFrequency('quality')
plotClasses = pl.PlotFrequency('coatingId')


# ### Define grid pipeline

# In[ ]:


grid = [
    loader,
    (
        ObservedData(),
        ImputedData()
    ),
    pl.SetYCol('coatingId'),
    pl.Set(scoring='f1_weighted'),
    pl.DropCol('coating'),
    (
        pl.AllValidFeatures(ignoreCols=ignoreXCols),
        pl.NonCollinearFeatures(keepCols=['teosVolPct', 'teosVolume'],
                             ignoreCols=ignoreXCols, show=False),
    ),
    pl.AugmentByQuality(F=2, scale=0.3, qcol='quality'),
    (
        None,
        pl.AugmentImb(RandomOverSampler()),
        pl.AugmentImb(BorderlineSMOTE()),
        pl.AugmentImb(SMOTE()),
        pl.AugmentImb(ADASYN()),
    ),
    # plotQuality,
    # plotClasses,
    pl.SplitValidation(),
    AggregateFeatures(show=False),
    pl.ScaleX(allColumns=False),
    (
        # SetModel(RandomForestClassifier()),
        pl.SetModel(DecisionTreeClassifier()),
    ),
    (
        None,
        pl.SelectFeaturesRFE(show=True)
    ),
    (
        pl.SetModel(XGBClassifier()),
        pl.SetModel(KNeighborsClassifier()),
        pl.SetModel(SVC()),
        pl.SetModel(GaussianProcessClassifier()),
        pl.SetModel(RandomForestClassifier()),
    ),
    pl.SearchHyperParams(hyperparams.space),
    TestPerformance(show=True)
]


# In[ ]:


pipe = pipeline.GridLine(grid)
pipe.Execute(inputCsv)


# In[ ]:


res = pipe.Summarize()
print(res)
try:
    res.to_csv("gridline_results.csv")
except:
    input("Please close the excel file if open and press enter ...")
    res.to_csv("gridline_results.csv")
    print("Saved")


# In[ ]:


class get_ipython:
    def system(*args):
        pass


# ```bash
# ## RUN THIS NOTEBOOK FROM THE TERMINAL
# $ jupyter nbconvert --to script PlayGround.ipynb
# $ python PlayGround.py
# ```
