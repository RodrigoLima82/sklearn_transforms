#import pandas as pd
#import numpy as np
#from imblearn.over_sampling import SMOTE
#from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class LabelEncode(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()

        non_features = self.columns

        le = LabelEncoder()

        for columns in non_features:
            fe_labels = le.fit_transform(data[columns])
            data[columns] = fe_labels
            #fe_mappings = {index: label for index, label in enumerate(le.classes_)}

        return data    

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        return data.drop(labels=self.columns, axis='columns')
       