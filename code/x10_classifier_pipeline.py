#!/usr/bin/env python
# coding: utf-8

# ```bash
# ## TO RUN THIS NOTEBOOK FROM THE TERMINAL
# $ jupyter nbconvert --to script x10_classifier_pipeline.ipynb
# $ python x10_classifier_pipeline.py
# ```

# In[1]:


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


# ### Common libraries

# In[2]:


import pipeline as pl
from model import plotlib


# In[3]:


from model.AdData import *
from model.AdFeatures import *
from model.AdClassify import TestPerformance
from model import hyperparams


# In[4]:


from importlib import reload
reload(pl)


# ### Initialize

# In[5]:


plotlib.load_fonts("../../../common/fonts/")
plt.style.use("matplotlib.mplstyle")

inputCsv = "../Data/imputed_data.mice.csv"
ignoreXCols = ['imp', 'id', 'quality', 'lobe', 'full', 'other', 'coatingId']


# In[6]:


loader = LoadData()
loader.Execute(inputCsv)


# ### Define grid pipeline

# In[7]:


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
        # SetModel(XGBClassifier()),
        pl.SetModel(KNeighborsClassifier()),
        # SetModel(SVC()),
        # SetModel(GaussianProcessClassifier()),
        # SetModel(RandomForestClassifier()),
    ),
    pl.SearchHyperParams(hyperparams.space),
    TestPerformance(show=True)
]


# In[8]:


reload(pl)
pipe = pipeline.GridLine(grid)
pipe.Execute(inputCsv)


# In[10]:


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
