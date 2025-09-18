#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""New library for HI-VAE

@author: athro
"""

# adepted class to have a general bi-directional transformer for categorical columns
# orginal idea from https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn



import pandas as pd
import sklearn
import sklearn.preprocessing

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns      = columns # array of column names to encode
        self.transformers = {}
        
    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                le = sklearn.preprocessing.LabelEncoder()
                output[col] = le.fit_transform(output[col])
                self.transformers[col] = le
        else:
            for colname,col in output.iteritems():
                le = sklearn.preprocessing.LabelEncoder()
                output[colname] = le.fit_transform(col)
                self.transformers[col] = le

        return output

    def inverse_transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                le = self.transformers[col]
                output[col] = le.inverse_transform(output[col])
        else:
            for colname,col in output.iteritems():
                le = self.transformers[col]
                output[colname] = le.inverse_transform(col)

        return output
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


if __name__ == '__main__':
    fruit_data = pd.DataFrame({
        'fruit':  ['apple','orange','pear','orange'],
        'color':  ['red','orange','green','green'],
        'weight': [5,6,3,4]
    })
    print(fruit_data)

    mcl = MultiColumnLabelEncoder(columns = ['fruit','color'])

    out = mcl.fit_transform(fruit_data)
    print(out)
    print(mcl.inverse_transform(out))
