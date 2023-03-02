import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')

def shap_forces(ml, df):
    explainer = shap.TreeExplainer(ml.model)

    # show the training distribution so we know
    # how to connect the scales
    df[ml.ytr.columns].plot.hist()
    ml.ytr.plot.hist()
    plt.show()

    for i in range(df.shape[0]):
        xrow = ml.Prep(df.iloc[i:i+1])
        orow = ml.Prep(df.iloc[i:i+1], False)
        xrow = xrow[ml.Xtr.columns]
        orow = orow[ml.Xtr.columns]
        shap_values = explainer.shap_values(xrow)
        shap.initjs()
        display(shap.force_plot(explainer.expected_value, shap_values, orow))
        print("Prediction = ", ml.Predict(xrow))

def shap_max(ml, df, explainer):
    xdf = ml.Prep(df)
    xdf = xdf[ml.Xtr.columns]
    shap_values = explainer.shap_values(xdf)
    return pd.Series(np.abs(shap_values).max(axis=0),
                     index=xdf.columns).sort_values(ascending=False)

def shap_summary(ml, df):
    xdf = ml.Prep(df)
    odf = ml.Prep(df, False)
    xdf = xdf[ml.Xtr.columns]
    odf = odf[ml.Xtr.columns]
    explainer = shap.TreeExplainer(ml.model)
    shap_values = explainer.shap_values(xdf)
    shap.initjs()
    display(shap.force_plot(explainer.expected_value, shap_values, odf))
    display(shap.summary_plot(shap_values, xdf))
    
    
# Total Main and Interaction between features.
# The coloring column is automatically chosen based on
# what might be the potential interaction. This can be specified though.
# Here every dot is a data point/row.
# The vertical dispersion at a single feature value results from interaction.

def shap_dependence(ml, df, xcol):
    xdf = ml.Prep(df)
    odf = ml.Prep(df, False)
    xdf = xdf[ml.Xtr.columns]
    odf = odf[ml.Xtr.columns]
    explainer = shap.TreeExplainer(ml.model)
    shap_values = explainer.shap_values(xdf)
    fig, ax = plt.subplots()
    shap.dependence_plot(xcol, shap_values, xdf, cmap='jet',
                         display_features=odf, ax=ax, show=False)
    plt.show()
    
    
def shap_interactions(ml, df, xcol):
    xdf = ml.Prep(df)
    odf = ml.Prep(df, False)
    xdf = xdf[ml.Xtr.columns]
    odf = odf[ml.Xtr.columns]
    explainer = shap.TreeExplainer(ml.model)
    shap_values = explainer.shap_interaction_values(xdf)

    # Main + Interaction Effects
    for ycol in xdf.columns:
        fig, ax = plt.subplots()
        shap.dependence_plot((xcol, ycol), shap_values, xdf, cmap='jet',
                         display_features=odf, ax=ax, show=False)
        plt.show()

import xgboost as xgb

def ShapXGBBestFeatures(ml, df, xgparams, K = 10, cutoff = 0.20):
    print(f"Calculating XGB {K}-fold CV SHAP values.")
    shVals = pd.DataFrame(columns = list(range(K)), index = ml.Xtr.columns)
    for i in range(K):
        ml.Split(ml.ytr.columns[0], ml.Xtr.columns)
        ml.Fit(xgb.XGBRegressor(**xgparams))
        shVals[i] = shap_max(ml, df, shap.TreeExplainer(ml.model))
        print("------------- K = %02d OK" %(i+1))

    return process_feats("XGB", ml, K, cutoff, shVals)

def process_feats(name, ml, K, cutoff, shVals, save = False):
    # Process and Plot Results
    shVals['Avg. SHAP'] = shVals.mean(axis=1)
    shVals['se'] = shVals.std(axis=1)
    shVals = shVals.sort_values('Avg. SHAP', ascending = False)

    fig, ax = plt.subplots(dpi=100, figsize=(5, shVals.shape[0] * 0.18))
    shVals[['Avg. SHAP', 'se']].plot.barh(xerr = 'se', color='r', capsize=4, ax = ax)
    ax.axvline(x=cutoff, linestyle = "--", color = "b")
    plt.title(f"SHAP {K}-fold CV for y = {ml.ytr.columns[0]} ({name})")
    plt.xlabel("Max Absolute SHAP value")
    if save:
        plt.savefig(save, dpi=600)
    else:
        plt.show()

    print(shVals[shVals['Avg. SHAP'] > cutoff]['Avg. SHAP'])
    
    # Best Features
    doecols = shVals[shVals['Avg. SHAP'] > cutoff].index
    return sorted(list(doecols))
