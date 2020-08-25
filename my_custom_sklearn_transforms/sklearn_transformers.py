import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class NewColumns(BaseEstimator, TransformerMixin):
    
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):              
        #X[self.column+'_1'] = np.sqrt(X[[self.column]].sum(axis=1))
        #X[self.column+'_2'] = np.tan(np.sqrt(X[[self.column]].sum(axis=1)))
        
        # Defini as colunas que serao agregadas
        cols   =  [['NOTA_DE'],['NOTA_EM'],['NOTA_MF'],['NOTA_GO']]
        target = 'PERFIL'

        # Percorre as colunas da lista para agregacao
        for col in cols:
            col_name = '_'.join(col)+'_'
            X[col_name + 'mean'] = X.groupby(col)[target].transform('mean').astype(np.float16)
            X[col_name + 'median'] = X.groupby(col)[target].transform('median').astype(np.float16)
            X[col_name + 'max'] = X.groupby(col)[target].transform('max').astype(np.float16)
            X[col_name + 'min'] = X.groupby(col)[target].transform('min').astype(np.float16)
            X[col_name + 'std'] = X.groupby(col)[target].transform('std').astype(np.float16)
            X[col_name + 'range'] = X[col_name + 'max'] - X[col_name + 'min']
            X[col_name + 'skew'] = X.groupby(col)[target].transform('skew').astype(np.float16)
            X[col_name + 'mad'] = X.groupby(col)[target].transform('mad').astype(np.float16)
            X[col_name + 'q25'] = X.groupby(col)[target].transform(lambda x: x.drop_duplicates().quantile(0.25)).astype(np.float16)
            X[col_name + 'q75'] = X.groupby(col)[target].transform(lambda x: x.drop_duplicates().quantile(0.75)).astype(np.float16)
            X[col_name + 'q95'] = X.groupby(col)[target].transform(lambda x: x.drop_duplicates().quantile(0.95)).astype(np.float16)
            X[col_name + 'iqr'] = X[col_name + 'q75'] - dados[col_name + 'q25']                

        return X
    
class SetIndex(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.set_index(self.columns, inplace=True)
