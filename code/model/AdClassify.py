import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from . import plotlib
import pipeline
from pipeline import utils


class TestPerformance(pipeline.Adapter):
    """ Test the performance of a classifier. """
    def __init__(self, savePref=None, show=True, use_validation = False, use_test = True):
        self.savePref = savePref
        self.show = show
        self.useValidationSet = use_validation

    def _plot_confusion(self, y, p, pl):
        if not self.savePref:
            return
        cf = confusion_matrix(y, p)
        f1 = classification_report(y, p, output_dict = True, zero_division=0)
        stat = "\n\nWeighted f1-score: %0.3f" %f1['weighted avg']['f1-score']
        
        name = pipeline.nice_name(pl.model)

        # Plot
        plotlib.make_confusion_matrix(cf, categories = pl.model.classes_,
                                      title = name,
                                      sum_stats = False,
                                      other_stat = stat)
        plt.minorticks_off()
        plt.tight_layout()
        if self.savePref:
            path = "%s.pipeline%02d.png" %(self.savePref, self.lineId)
            plt.savefig(path)
            print("Save OK:", path)
        if self.show:
            plt.show()
        else:
            plt.close()
            
    def _calc_score(self, y, p, pl):
        f1 = classification_report(y, p, output_dict = True, zero_division=0)
        
        criteria = [
            ('100% accuracy', f1['weighted avg']['f1-score'] == 1),
            ('5/6 accuracy',  f1['weighted avg']['f1-score'] > 5/6),
        ]
        
        # per class accuracy
        clsnames = pl.Tr[pl.yCol].unique()
        for c in clsnames:
            c = str(c)
            r = "100%% accuracy on %s" %c
            v = False
            if c in f1:
                v = f1[c]['f1-score'] >= 0.99
            criteria.append((r, v))
            
            r = "80%% accuracy on %s" %c
            v = False
            if c in f1:
                v = f1[c]['f1-score'] >= 0.80
            criteria.append((r, v))
            
        # make data frame of the results
        pl.score_report = pd.DataFrame({ 'result': [i[1] for i in criteria] },
                                    index = [i[0] for i in criteria])
        pl.score = pl.score_report.sum().values[0]
        pl.score = f1['2']['f1-score']
        
        if self.useValidationSet:            
            pl.stats['val_wt_f1'] = f1['weighted avg']['f1-score']
            pl.stats['val_acc'] = f1['accuracy']
        else:
            pl.stats['wt_f1'] = f1['weighted avg']['f1-score']
            pl.stats['acc'] = f1['accuracy']

        return pl

    def Process(self, pl):
        if self.useValidationSet:
            assert pl.Tv is not None, "Validation set not defined"
            X = pl.Tv[pl.xCols]
            y = pl.Tv[pl.yCol]
            self.sayf("Run on validation dataset.")
        else:
            assert pl.Ts is not None, "Test set not defined"
            X = pl.Ts[pl.xCols] # use the test set
            y = pl.Ts[pl.yCol]

        p = pl.model.predict(X)
        pl = self._calc_score(y, p, pl)
        self.sayf("SCORE: {}", pl.score)
        if self.show or pl.score > 0:
            print()
            print(pl.score_report)
            self.sayf("{}", classification_report(y, p, zero_division=0))
        self._plot_confusion(y, p, pl)
        return pl
