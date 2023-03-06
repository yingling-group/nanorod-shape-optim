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


# In[ ]:


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import kernels, GaussianProcessClassifier


# ### Load libraries

# In[32]:


# Custom scikit-learn like pipeline with additional functionalities
import pipeline as pl
import pipeline


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

runName = 'classifier_pl_run1800'
outputCsv = "%s_results.csv" %runName


# In[ ]:


# Save outputs to log files
pl.set_stderr("%s.errlog.txt" %runName, fout="%s.log.txt" %runName)


# In[ ]:


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


class RunCond(pl.Adapter):
    """ Run an adapter if condition passes. """
    def __init__(self, adapter_instance):
        self.adpt = adapter_instance

    def __repr__(self):
        return "RunCond: " + pl.nice_name(self.adpt)
    
    def Process(self, pl):
        if pipeline.nice_name(pl.model) in ["GaussianProcessClassifier"]:
            self.sayf("{} skipped, run condition failed.", pipeline.nice_name(self.adpt))
            return pl
        
        return self.adpt.Execute(pl)


# In[ ]:


grid = [
    loader,
    (
        ObservedData(),
        ImputedData()
    ),
    pl.SetYCol('coatingId'),
    pl.Set(scoring='f1_weighted'), #scoring used by sklearn
    pl.DropCol('coating'),
    (
        pl.AllValidFeatures(ignoreCols=ignoreXCols),
        pl.NonCollinearFeatures(keepCols=['teosVolPct', 'teosVolume'],
                             ignoreCols=ignoreXCols, show=False),
    ),
    (
        None,
        pl.AugmentByQuality(F=1, scale=0.3, qcol='quality'),
        pl.AugmentByQuality(F=2, scale=0.4, qcol='quality'),
    ),
    (
        None,
        pl.ScaleX(allColumns=False),
        pl.ScaleX(allColumns=True)
    ),
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
    (
        None,
        AggregateFeatures(show=False)
    ),
    pl.ScaleX(allColumns=True),
    pl.SetModel(DecisionTreeClassifier()),
    pl.SelectFeaturesRFE(show=True),
    (
        pl.SetModel(XGBClassifier()),
        pl.SetModel(KNeighborsClassifier()),
        pl.SetModel(SVC()),
        pl.SetModel(GaussianProcessClassifier()),
        # pl.SetModel(RandomForestClassifier()),
    ),
    RunCond(pl.SearchHyperParams(hyperparams.space)),
    TestPerformance(show=True, use_validation=True),
    TestPerformance(show=True, use_test=True),
]


# In[ ]:


pipe = pipeline.GridLine(grid)
pipe.Shuffle() # suffle the list, so we immediately have data for analysis
pipe.Save(outputCsv) # save the results after each pipeline run


# In[ ]:


pipe.grid


# In[ ]:


pipe.Execute(inputCsv)
res = pipe.Summarize()
print(res)


# In[ ]:


try:
    res.to_csv(outputCsv)
except:
    input("Please close %s if open and press enter ..." %outputCsv)
    res.to_csv(outputCsv)
    print("Save OK:", outputCsv)

