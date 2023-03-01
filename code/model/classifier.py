import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from . import pipeline
from . import plotlib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

class Classifier(pipeline.Pipeline):
    """ A classifier class to handle pandas DataFrames and maintain the train/test pipeline """

    def __init__(self, df = None):
        super(Classifier, self).__init__(df)
    
    def Report(self, Ts):
        Ts = self._prep_df(Ts)
        assert self.yCol in Ts.columns, "Ts does not contain the response"

        y = Ts[self.yCol]
        p = self.Predict(Ts)
        print(classification_report(y, p, zero_division=0))
        
        # Numeric ycol needed for AUC
        if self.yCol in Ts.select_dtypes(include=['int64','float64']):
            print("AUC = ", roc_auc_score(y, p))

    def Confusion(self, Ts, savePrefix = None, show = True):
        """ Plot confusion matrix of the test dataset """

        Ts = self._prep_df(Ts)
        assert self.yCol in Ts.columns, "Ts does not contain the response"

        y = Ts[self.yCol]
        p = self.Predict(Ts)
        cf = confusion_matrix(y, p)
        f1 = classification_report(y, p, output_dict = True, zero_division=0)
        stat = "\n\nWeighted f1-score: %0.3f" %f1['weighted avg']['f1-score']

        # Plot
        plotlib.make_confusion_matrix(cf, categories = self.model.classes_,
                                      title = self.Name,
                                      sum_stats = False,
                                      other_stat = stat)
        plt.minorticks_off()
        plt.tight_layout()
        if savePrefix:
            path = "%s.%s.png" %(savePrefix, self.Name)
            plt.savefig(path)
            print("Save OK:", path)
        if show:
            plt.show()
        else:
            plt.close()


def New(df, xcols, ycol, fnlist):
    clf = Classifier(df)
    clf.AddFeatures(fnlist, show_list=False)
    clf.SetColumns(xcols, ycol)
    return clf
