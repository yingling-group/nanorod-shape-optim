import numpy as np 
import pandas as pd

class dfBuilder:
    """ Build a pandas dataframe """
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

    @property
    def df(self):
        return pd.DataFrame(self.dict)
    
    def __repr__(self):
        return str(self.df)
