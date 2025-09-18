#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""New library for HI-VAE

@author: athro
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import pandas as pd
#from hivae import hivae
import hivae #.hivae as hivae

import warnings

class HIVAEError(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message

from hivae.preprocessing import MultiColumnLabelEncoder

HiVAE_cat_types = ['cat','ordinal','ord']

def get_basic_label_encoding(df_sub,column_names,use_columns_categories):
    col_cats_raw = set(column_names).intersection(set(use_columns_categories.keys()))
    cols_cat = [x for x in column_names if x in col_cats_raw]
    
    mcl = MultiColumnLabelEncoder(cols_cat)
    mcl.fit(df_sub)
    
    return mcl

class HiVAEImputer(BaseEstimator, TransformerMixin):
    '''
    Class used for imputing missing values in a pd.DataFrame using hivae.
    
    Parameters
    ----------    
    use_columns : list
        List of columns to be used for imputation
    impute_columns : list
        List of columns to be imputed - if lecft empty [] - all use_columns will be used (apart from target columns) 
    target_columns : list
        List of columns considered a target variable (y) - these will not be employed in imputing
    use_columns_categories : None
        Note sure what that was for
    epochs: integer (10)
    batch_size:  32 # maxed to number of samples
    model_name: 'model_HIVAE_inputDropout','model_HIVAE_factorized'
    dim_z: 12
    dim_y: 5
    dim_s=: 10
    dir_hivea       = './hivae'
    verbosity_level = 3

    Returns
    -------
    X : array-like
        The array with imputed values in the target column
    '''

    batch_size  = 32 
    model_name  = 'model_HIVAE_inputDropout'
    dim_z       = 12
    dim_y       = 5
    dim_s       = 10
    epochs      = 0
    cols_use    = []
    target      = None
    
    def __init__(self, 
                 use_columns, 
                 impute_columns  = [], 
                 target_columns  = [], 
                 use_columns_categories    = None, 
                 epochs          = 10,
                 batch_size      = 32,
                 model_name      = 'model_HIVAE_inputDropout',
                 dim_z           = 12,
                 dim_y           = 5,
                 dim_s           = 10,
                 #network_def    = None,
                 dir_hivea       = './hivae',
                 verbosity_level = 0,
                ):
        
        # ensuring minial compliance
        #assert type(use_columns) == list, 'use_columns should be a list of columns - used for building model'
        #assert type(target_columns) == list, 'target_columns' hould be a list of column -  where to impute'
        #assert type(target) == str, 'target should be a string'

        
        # setting hivae verbosity level
        self.verbosity_level = verbosity_level
        
        # where to save intermediate results and models
        # that might be changed to a temp directory if not specfied
        self.dir_hivea    = dir_hivea
        # where the results will be saved
        self.results_path = '{}/results/'.format(dir_hivea)
        # where the networks will be saved
        self.network_path = '{}/network/'.format(dir_hivea)
        
        # save network definitions
        #if network_def == None:
        #    self.network_def = self._guess_network_def(cols_use)

        # save the supplied information
        
        self.batch_size     = batch_size
        self.model_name     = model_name
        self.dim_z          = dim_z
        self.dim_y          = dim_y
        self.dim_s          = dim_s
        #self.network_def    = network_def
        self.epochs         = epochs

        self.target_columns = target_columns
        # do not use target columns in any way
        # not allowed in __init__
        self.use_columns    = use_columns # [x for x in use_columns if x not in target_columns]
        if impute_columns == []:
            self.impute_columns = self.use_columns
        else:
            self.impute_columns = impute_columns
        
        # autogenerate generate_type_list?
        #    raise HIVAEError({'message':'automatic type_list generatation not yet implemented'})
        if use_columns_categories == None or use_columns_categories=={}:
            self.use_columns_categories = {}
        else:
            self.use_columns_categories = use_columns_categories
        #self.target = target

        # create unique string for this setting
        id_hex =  '0x{:x}'.format(id(self))
        id_hash = hash(f'{self.batch_size}-{self.model_name}-{self.dim_z}-{self.dim_y}-{self.dim_s}-{self.epochs}-{self.use_columns}-{self.impute_columns}-{self.target_columns}')
        self.string_hash = f'{id_hex}__{id_hash}'

#    @property
#    def verbosity(self):
#        return self.
        
    def __repr__(self):
        return '{}(attrib={},hash={})'.format(self.__class__.__name__,len(self.use_columns),self.string_hash)
    
    def _encode_data(self,df_local):
        
        df_ana_local              = df_local[self.use_columns]

        
        self.multi_encoder        = get_basic_label_encoding(df_ana_local,self.use_columns,self.use_columns_categories) 
        df_ana_local_encoded_raw  = self.multi_encoder.transform(df_ana_local)
        
        df_ana_local_missing_mask = self.__generate_missing_mask(df_ana_local,self.use_columns)

        
        df_ana_local_encoded      = df_ana_local.where(df_ana_local_missing_mask==0, df_ana_local_encoded_raw)
        #df_ana_local_encoded.dtypes

        df_ana_local_encoded      = df_ana_local_encoded.astype('float')
        return df_ana_local_encoded,df_ana_local_missing_mask

        
    
    # def _guess_network_def(self,cols_use):
    #     network_def = network_def = {
    #         'batch_size' : 32,
    #         'model_name': 'model_HIVAE_inputDropout',
    #         'dim_z': int(len(cols_use)/2+1), # embedding (bottleneck)
    #         'dim_y': 5,
    #         'dim_s': 10,
    #         }
    #     return network_def
   
    
    def fit(self, X, y=None):

        #print('hivae.fit1','type(X)', type(X))
        
        # nneds to be done here - not allowed in __init__
        self.use_columns    = [x for x in self.use_columns if x not in self.target_columns]

        
        #assert pd.isnull(X[self.group_cols]).any(axis=None) == False, 'There are missing values in group_cols'
        df_ana_local         = X.copy()[self.use_columns]
        #print('hivae.fit',df_ana_local.shape)
        types_list_local     = self.generate_type_list(df_ana_local,self.use_columns_categories,self.use_columns)

        df_ana_local_encoded,df_ana_local_missing_mask = self._encode_data(df_ana_local)
        #print(df_ana_local_encoded.head())
        #print(df_ana_local_missing_mask.head())

        
        #self.generate_type_listprint('network_def = <<{}>>'.format(self.network_def))
        if len(X) < self.batch_size:
            hivae.hivae.vprint_s(2,'batch_size has been adepted from {} to {} because of dataset size'.format(self.batch_size,len(X)))
            self.batch_size     = min(self.batch_size,len(X))
        local_network_def = {
            'batch_size' : self.batch_size,
            'model_name':  self.model_name,
            'dim_z':       self.dim_z, # embedding (bottleneck)
            'dim_y':       self.dim_y,
            'dim_s':       self.dim_s,
            }

        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #hivae_obj = hivae.hivae(
            hivae_obj = hivae.hivae(
                types_list_local,
                local_network_def,
                #results_path = self.results_path,
                #network_path = self.network_path
                )
        
            hivae.hivae.set_verbosity_level_s(self.verbosity_level)
            
            
            hivae_obj.fit(
                df_ana_local_encoded,
                epochs=self.epochs,
                true_missing_mask=df_ana_local_missing_mask)

        #impute_map = X.groupby(self.group_cols)[self.target].agg(self.metric).reset_index(drop=False)

        #print('hivae.fit2 -hivae_obj', hivae_obj)
        
        self.hivae_obj_ = hivae_obj
        #self.impute_map_ = impute_map
        
        return self 
    
    def transform(self, X, y=None):
        
        # make sure that the imputer was fitted
        check_is_fitted(self, 'hivae_obj_')

        # in case verbosity level has changed - causes some problems
        #self.hivae_obj_.set_verbosity_level(self.verbosity_level)
        
        df_transform  = X.copy()[self.use_columns]

        df_transform_encoded,df_transform_missing_mask = self._encode_data(df_transform)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            (test_data, test_data_reconstructed, test_data_decoded, test_data_embedded_z, test_data_embedded_s) = self.hivae_obj_.predict(df_transform_encoded,true_missing_mask=df_transform_missing_mask)

        # into dataframes
        #df_test_data   = pd.DataFrame(test_data,columns=df_encoded.columns,index=df_encoded.index)
        df_test_data_reconstructed = pd.DataFrame(test_data_reconstructed,columns=df_transform_encoded.columns,index=df_transform_encoded.index)
        df_test_data_reconstructed[self.multi_encoder.columns] = df_test_data_reconstructed[self.multi_encoder.columns].astype(int)

        
        df_transform_filled = self.multi_encoder.inverse_transform(df_test_data_reconstructed)

        df_return_filled  = X.copy()
        # only copy the columns required
        df_return_filled[self.impute_columns] = df_transform_filled[self.impute_columns]

        return df_return_filled
        
        #X = X.copy()
        
        #for index, row in self.impute_map_.iterrows():
        #    ind = (X[self.group_cols] == row[self.group_cols]).all(axis=1)
        #    X.loc[ind, self.target] = X.loc[ind, self.target].fillna(row[self.target])
        
        #return X.values
        #return df_transform_filled
        #return df_transform_filled.values
    
    def __generate_missing_mask(self,df,features_use):
        df_missing_mask  = pd.DataFrame(columns=features_use,index=df.index)
        for x in features_use:
            df_missing_mask[x] = df[x].isna().map({True:0,False:1})
        return df_missing_mask

    # needs sorting out later
    #def set_verbosity_level(self,set_verbosity_level=0):
    #    if set_verbosity_level>=0 and set_verbosity_level<=4:
    #        self.set_verbosity_level = set_verbosity_level
    
    #@classmethod
    def generate_type_list(self,df_sub,column_names_categorical,column_names=None):
        types_list = []
        if column_names == None:
                column_names = self.use_columns_categories.keys
        for feature_name in column_names:
            if feature_name in self.use_columns_categories.keys():
                data_type = self.use_columns_categories[feature_name]
                if data_type in ['cat','categorical']:
                    # ther might be a problem as training data does not contain all categories
                    num_cats = len(df_sub[feature_name].value_counts(dropna=True).index) 
                    types_list += [(attrib,'cat',num_cats,num_cats)] 
                elif data_type in ['ord','ordinal']:
                    num_ords = len(df_sub[feature_name].value_counts(dropna=True).index) 
                    types_list += [(attribfeature_name,'ordinal',num_ords,num_ords)] 
                elif data_type in ['pos']:
                    types_list += [(feature_name,'pos',1,None)]
                elif data_type in ['real']:
                    types_list += [(feature_name,'real',1,None)]
                elif data_type in ['count']:
                    types_list += [(feature_name,'count',1,None)]
                else:
                    raise HIVAEError({'message':'Categorical variables supplied, but not in use_columns'})
        return types_list
        

# class ColumnSelector(BaseEstimator, TransformerMixin):
    
#     def __init__(self, dtype):
#         self.dtype = dtype
    
#     def fit(self, X, y=None):
#         """ Get either categorical or numerical columns on fit.
#         Store as attribute for future reference"""
#         X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
#         if self.dtype == 'numerical':
#             self.cols = X.select_dtypes(exclude='O').columns.tolist()
#         elif self.dtype == 'categorical':
#             self.cols = X.select_dtypes(include='O').columns.tolist()
#         self.col_idx = [df.columns.get_loc(col) for col in self.cols]
#         return self

#     def transform(self, X):
#         """ Subset columns of chosen data type and return np.array"""
#         X = X.values if isinstance(X, pd.DataFrame) else X
#         return X[:, self.col_idx]
        
        
# class GeneralImputer(BaseEstimator, TransformerMixin):
#     imput_obj = None
#     def __init__(self,imput_obj=None):
#         self.imput_obj = imput_obj
#         print('init',self.__class__.__name__,self.imput_obj)
#     def fit(self, *args):
#         if self.imput_obj:
#             print('fit',self.__class__.__name__,self.imput_obj.__repr__)
#             self.imput_obj.fit(*args)
#     def transform(self, *args):
#         if self.imputer:
#             print('transform',self.__class__.__name__,self.imput_obj.__repr__)
#             return self.imput_obj.transform(*args)
