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


from model import plotlib
from model import pipeline
from model import utils
from model.AdCommon import *
from model.AdFeature import *
from model.AdAugment import *
from model.AdHyperParam import *
from model.AdScaler import *


# In[3]:


from model.AdClassify import *


# In[4]:


from importlib import reload
reload(pipeline);


# ### Project libraries

# In[5]:


from uv_vis.AdData import LoadData, ObservedData, ImputedData
from uv_vis.AdFeatures import AggregateFeatures
from uv_vis import hyperparams


# ### Initialize

# In[6]:


plotlib.load_fonts("../../../common/fonts/")
plt.style.use("matplotlib.mplstyle")

inputCsv = "../Data/imputed_data.mice.csv"
ignoreXCols = ['imp', 'id', 'quality', 'lobe', 'full', 'other', 'coatingId']


# In[7]:


loader = LoadData()
loader.Execute(inputCsv)


# ### Define grid pipeline

# In[8]:


grid = [
    loader,
    (
        ObservedData(),
        ImputedData()
    ),
    SetYCol('coatingId'),
    Set(scoring='f1_weighted'),
    DropCol('coating'),
    AugmentByQuality(F=2, scale=0.3, qcol='quality'),
    (
        None,
        AugmentImb(RandomOverSampler()),
        AugmentImb(BorderlineSMOTE()),
        AugmentImb(SMOTE()),
        AugmentImb(ADASYN()),
    ),
    AggregateFeatures(show=False),
    (
        AllValidFeatures(ignoreCols=ignoreXCols),
        NonCollinearFeatures(keepCols=['teosVolPct', 'teosVolume'],
                             ignoreCols=ignoreXCols, show=False),
    ),
    ScaleX(allColumns=False),
    (
        # SetModel(RandomForestClassifier()),
        SetModel(DecisionTreeClassifier()),
    ),
    (
        None,
        SelectFeaturesRFE(show=True)
    ),
    (
        # SetModel(XGBClassifier()),
        SetModel(KNeighborsClassifier()),
        # SetModel(SVC()),
        # SetModel(GaussianProcessClassifier()),
        # SetModel(RandomForestClassifier()),
    ),
    SearchHyperParams(hyperparams.space),
    TestPerformance(show=True)
]


# In[9]:


reload(pipeline)
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
