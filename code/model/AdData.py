import numpy as np
import pandas as pd

import pipeline

class LoadData(pipeline.Adapter):
    """ Load and train/test split the data """
    def __init__(self, resample = False):
        self.testdf = None
        self.traindf = None
        # if train test split should be performed for each line
        self.resample = resample

    def add_classes(self, data):
        data = data.assign(coating = data[['lobe', 'full', 'other']].idxmax(axis=1))
        data = data.assign(coatingId = data.coating.replace(data.coating.unique(),
                                                    range(len(data.coating.unique()))))
        data = data.set_index('name')
        self.sayf("Add coating classes: {}", list(data.coating.unique()))
        return data
    
    def calc_train_test(self, data):
        while True:
            # Sample 6 data points for testing
            testIds = data[data.imp == 0].dropna().id.sample(n=6)

            # Separate test and training data
            self.testdf = data[data.imp == 0].loc[lambda df: df.id.isin(testIds)]
            self.traindf = data.loc[lambda df: ~df.id.isin(testIds)].dropna()

            # keep only 1 lobe and all three classes
            if list(self.testdf.coating).count('lobe') == 1 and len(self.testdf.coating.unique()) == 3:
                break

        self.sayf("Test IDs: {}", list(testIds))
        self.sayf("Test classes: {}", list(self.testdf.coating))

    def Process(self, csv):
        if self.testdf is None:
            data = pd.read_csv(csv)
            data = self.add_classes(data)
            self.calc_train_test(data)
        elif self.resample:
            self.calc_train_test(data)

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
    
