import numpy as np 
import pandas as pd
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor

class dfBuilder:
    """ Build a pandas dataframe one row at a time. """
    def __init__(self):
        self.dict = {}
        self.rows = 0

    def add(self, **kwargs):
        for col in kwargs:
            if col in self.dict:
                self.dict[col].append(kwargs[col])
            else:
                self.dict[col] = [''] * self.rows
                self.dict[col].append(kwargs[col])
        self.rows += 1
        # print(kwargs)

    @property
    def df(self):
        return pd.DataFrame(self.dict)
    
    def __repr__(self):
        return str(self.df)


def summarize_results(res, groupBy, imputeCol = None, ignoreValues = None, includeIndividual=True):
    """ Summarize the results of K-fold CV. 
        If imputeCol is given, also summarize the results of imputed data and using ignoreValues.
    """
    completeSummary = res.groupby(groupBy).agg([np.mean, np.std])
    completeSummary.columns = completeSummary.columns.map('|'.join).str.strip('|')
    completeSummary = completeSummary.reset_index()

    if imputeCol is None:
        return completeSummary
    else:
        assert ignoreValues is not None, "value for the complete observations needed in column '%s'" %imputeCol
        imputeMask = res[imputeCol] != ignoreValues
        df = res[imputeMask]

        if imputeCol in groupBy:
            groupBy.remove(imputeCol)

        # Per imputation summary
        summary = df.groupby(groupBy + [imputeCol]).agg([np.mean, np.var])
        summary.columns = summary.columns.map('|'.join).str.strip('|')
        summary = summary.reset_index()

        # variance prefactor for imputation
        m = summary.shape[0]
        prefactor = (1 + 1 / m)

        # mean value = average of the means
        # variance due to randomness = average of the variances
        means = summary.groupby(groupBy).mean(numeric_only=True)

        # variance due to imputation = variance of the means
        variances = summary.groupby(groupBy).var(numeric_only=True)
        variances = variances.loc[:, variances.columns.str.endswith("|mean")]

        # Update total mean and variance due to imputation
        # Change column names to |var to align with the variance columns
        variances.columns = [col.replace("|mean", "|var") for col in variances.columns]
        total = means.add(prefactor * variances, fill_value = 0)

        # Calculate standard deviations
        stdDev = total.loc[:, total.columns.str.endswith("|var")].apply(np.sqrt)
        stdDev.columns = [col.replace("|var", "|std") for col in stdDev.columns]
        total = pd.concat([total, stdDev], axis=1)

        # Remove the variance columns
        varCols = total.columns[total.columns.str.endswith("|var")]
        total = total.drop(columns=varCols)

        # Add name and sort column names as the original
        total[imputeCol] = "imputedObs"
        total = total.reset_index()

        # Other observations
        if includeIndividual:
            completeSummary = res.groupby(groupBy + [imputeCol]).agg([np.mean, np.std])
        else:
            completeSummary = res[~imputeMask].groupby(groupBy + [imputeCol]).agg([np.mean, np.std])
        completeSummary.columns = completeSummary.columns.map('|'.join).str.strip('|')
        completeSummary = completeSummary.reset_index()

        total = pd.concat([total, completeSummary], ignore_index=True)[completeSummary.columns]

        return total.sort_values(groupBy + [imputeCol])


def calc_vif(df):
    """ Calculate and return a DataFrame of VIF of the columns for the given df. """

    X = df.dropna().select_dtypes(include='number') #will include all the numeric types
    vif = pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['Feature'] = X.columns
    vif = vif.set_index('Feature').dropna()
    vif = vif.sort_values('VIF', ascending=False).T
    return vif.round(2)
