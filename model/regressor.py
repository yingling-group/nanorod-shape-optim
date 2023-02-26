import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from . import pipeline
from sklearn.metrics import r2_score, mean_squared_error

class Regressor(pipeline.Pipeline):
    """ A regressor class to handle pandas DataFrames and maintain the train/test pipeline """

    def __init__(self, df = None):
        super(Regressor, self).__init__(df)

    def Predict(self, df):
        assert isinstance(df, pd.DataFrame), f"df must be a DataFrame, not {type(df)}"
        assert self.model != None, "Please train a model first"
        for c in self.xCols:
            assert c in df.columns, \
                f"Column not found: {c}, please call PrepPrediction() first"

        X = df[self.xCols]
        X = self._scaleX(X)

        yp = self.model.predict(X)
        yp = pd.Series(yp, name=self.yCol, index = X.index)
        yp = self._unscaleY(yp)

        return yp
        
    def Fitness(self, Ts = None, save = False):
        """ Calculate and plot fitness of the given test dataset.
            If not dataset is given, use the splitted test dataset.
        """
        Ts = self._prep_df(Ts)
        assert self.yCol in Ts.columns, "Ts does not contain the response"

        Ts = self._prep_df(Ts)
        xtr = self.Tr[self.xCols]
        xts = Ts[self.xCols]
        ytr = self.Tr[self.yCol]
        yts = Ts[self.yCol]

        # fitness of the training data
        ptr = self.Predict(xtr)

        R2 = r2_score(ytr, ptr)
        MSE = mean_squared_error(ytr, ptr)

        # Plot
        fig, ax = plt.subplots(1, 3, figsize=(8, 2.5))
        ax[0].plot(ytr, ptr, 'bx')
        dline = np.linspace(*ax[0].get_xlim())
        ax[0].plot(dline, dline, 'k--')
        ax[0].set_xlabel("Prior predictor")
        ax[0].set_title("R2 = %0.2f, RMSE = %0.2f" %(R2, np.sqrt(MSE)))

        # fitness of the test data
        pts = self.Predict(xts)

        R2 = r2_score(yts, pts)
        MSE = mean_squared_error(yts, pts)

        ax[1].plot(yts, pts, 'bx')
        dline = np.linspace(*ax[1].get_xlim())
        ax[1].plot(dline, dline, 'k--')
        ax[1].set_xlabel("Posterior predictor")
        ax[1].set_title("R2 = %0.2f, RMSE = %0.2f" %(R2, np.sqrt(MSE)))
        
        ax[2].axhline(y=0, linestyle='--')
        ax[2].plot(yts, pts - yts, 'r.')
        ax[2].set_xlabel("Posterior residuals")

        plt.suptitle(self.Name)
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=600)
            plt.close(fig)
        else:
            plt.show()
            
        return {
            'model': self.Name,
            'R2': R2,
            'MSE': MSE
        }

    def ParityAndResidual(self, Ts = None, output = None):
        """ Plot parity and residuals of the given test dataset.
            If not dataset is given, use the splitted test dataset.
        """
        Ts = self._prep_df(Ts)
        assert self.yCol in Ts.columns, "Ts does not contain the response"

        # Scale
        y = Ts[self.yCol]
        p = self.Predict(Ts)

        R2 = r2_score(y, p)
        MSE = mean_squared_error(y, p)

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(6, 2.5), sharey=False)
        ax[0].plot(y, p, 'bx')
        dline = np.linspace(*ax[0].get_xlim())
        ax[0].plot(dline, dline, 'k--')
        ax[0].set_xlabel("Truth")
        ax[0].set_ylabel("Prediction")
        ax[0].set_title("R2 = %0.2f, RMSE = %0.2f" %(R2, np.sqrt(MSE)))

        ax[1].plot(y, p - y, 'r.')
        ax[1].axhline(y=0, linestyle='--')
        ax[1].set_xlabel("Residuals")

        plt.suptitle(self.Name)
        plt.tight_layout()
        if output:
            plt.savefig(output, dpi=600)
            plt.close()
            print("Save OK:", output)
        else:
            plt.show()
        return {
            'model': self.Name,
            'R2': R2,
            'MSE': MSE
        }


def New(df, xcols, ycol, fnlist):
    reg = Regressor(df)
    reg.SetColumns(xcols, ycol)
    reg.AddFeatures(fnlist)
    reg.Split()
    return reg
