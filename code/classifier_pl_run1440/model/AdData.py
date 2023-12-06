import numpy as np
import pandas as pd

import pipeline

class LoadData(pipeline.Adapter):
    """ Load and train/test split the data """
    def __init__(self, csv, resample = False, folds = 5):
        self.testdf = None
        self.traindf = None
        
        # if train test split should be performed for each line
        self.resample = resample
        self.folds = folds
        
        data = pd.read_csv(csv)
        self.labels = {
            "full": 0,
            "other": 1,
            "lobe": 2,
        }
        data = data.assign(coating = data[['lobe', 'full', 'other']].idxmax(axis=1))
        data = data.assign(coatingId = lambda df: [self.labels[n] for n in df.coating])
        self.data = data.set_index('name')
        
    
    def calc_train_test(self, data):
        while True:
            # Sample 6 data points for testing
            testIds = data[data.imp == 0].dropna().id.sample(n=6)

            # Separate test and training data
            self.testdf = data[data.imp == 0].loc[lambda df: df.id.isin(testIds)]
            self.traindf = data.loc[lambda df: ~df.id.isin(testIds)].dropna()

            # keep only 1 lobe and all three classes
            if list(self.testdf.coating).count('lobe') == 1 \
                and len(self.testdf.coating.unique()) == 3 \
                and self.testdf.index.str.startswith("Rowe").sum() == 1:
                break

        self.sayf("Test IDs: {}", list(testIds))
        self.sayf("Test classes: {}", list(self.testdf.coating))

    def Process(self, X = None):
        if self.resample or self.testdf is None:
            self.calc_train_test(self.data)
            self.sayf("Performed train/test split.")

        payload = pipeline.Payload()
        payload.Tr = self.traindf.copy()
        payload.Ts = self.testdf.copy()
        return payload
    
class ImputedData(pipeline.Adapter):
    """ Select imputed data as the training data """
    def Process(self, X):
        X.Tr = X.Tr[X.Tr.imp > 0]
        return X
    
class ObservedData(pipeline.Adapter):
    """ Select observed data as the training data """
    def Process(self, X):
        X.Tr = X.Tr[X.Tr.imp == 0]
        X.cv = 3
        return X
    
