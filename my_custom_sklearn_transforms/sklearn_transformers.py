#import pandas as pd
#import numpy as np
#from imblearn.over_sampling import SMOTE
#from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        return data.drop(labels=self.columns, axis='columns')
    
class SetIndex(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        return data.set_index(self.columns, inplace=True)
    
   
class CatEncode(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()

        non_features = self.features

        le = LabelEncoder()

        for features in non_features:
            fe_labels = le.fit_transform(data[features])
            data[features] = fe_labels
            fe_mappings = {index: label for index, label in enumerate(le.classes_)}

        return data    
