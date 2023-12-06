import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error
import pipeline


class TestPerformance(pipeline.Adapter):
    """ Test the performance of a regressor. """
    def __init__(self, savePref=None, show=True, use_validation = False, use_test = True):
        self.savePref = savePref
        self.show = show
        self.useValidationSet = use_validation

    def _plot_parity(self, y, p, pl):
        if not self.savePref:
            return
        
        R2 = r2_score(y, p)
        RMSE = mean_squared_error(y, p, squared=False)
        
        fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.5))
        ax.plot(y, p, 'rx')
        dline = np.linspace(*ax.get_xlim())
        ax.plot(dline, dline, 'k--')
        ax.set_xlabel("Truth")
        ax.set_ylabel("%s Prediction" %pipeline.nice_name(pl.model))
        ax.set_title("R2 = %0.2f, RMSE = %0.2f" %(R2, RMSE))

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
        R2 = r2_score(y, p)
        RMSE = mean_squared_error(y, p, squared=False)

        # data frame of the results
        pl.score_report = None
        pl.score = R2
        
        if self.useValidationSet:            
            pl.stats['val_r2'] = R2
            pl.stats['val_rmse'] = RMSE
        else:
            pl.stats['r2'] = R2
            pl.stats['rmse'] = RMSE

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
        self.sayf("R2 SCORE: {}", pl.score)
        if pl.score > 0.5:
            self._plot_parity(y, p, pl)
        return pl
