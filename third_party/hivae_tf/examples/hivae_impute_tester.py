#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numbers
import random 
import scipy.stats
import time
import shutil


# In[2]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))


# In[3]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import hivae
import hivae.impute
import hivae.preprocessing
#print(dir(hivae.impute))
#print(dir(hivae.preprocessing))


# In[4]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# create a fake dataset


# creare a simple train - test based dataset
# first get the IRIS demail?ata (all numerical) - apart from target (y)
X, y = load_iris(return_X_y=True)

# create randomly missing data
mask = np.random.randint(0, 2, size=X.shape).astype(bool)
X[mask] = np.nan

# split intro train test - for demonstration only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50,random_state=0)

# create as dataframes
column_names = [f'c{x}' for x in range(X_test.shape[1])]

df_train = pd.DataFrame(X_train,columns=column_names)
df_train['target'] = y_train
df_test = pd.DataFrame(X_test,columns=column_names)
df_test['target'] = y_test


display(df_train.head(10))



# In[5]:


use_columns = column_names
target_columns = ['target']

# create type of columns - here we set all to real
# all possibilityies are 
# 'real'        - real
# 'pos'         - positive real only
# 'count'       - positive integer values
# 'cat'         - categorical 
# 'categorical' - categorical --- not sure if implemented troughout
# 'bool'        - categorical --- not sure if implemented troughout
# 'ord'         - ordered categorical
# 'ordinal'     - ordered categorical
use_columns_categories = dict(zip(use_columns,len(use_columns)*['real']))
display(use_columns_categories)

# define which colums to impute - if [] all columns will be imputed
impute_columns = ['c0','c3'] 


# In[6]:


# create imputer object:

hi = hivae.impute.HiVAEImputer(use_columns,
                               impute_columns=impute_columns,target_columns=target_columns,
                               use_columns_categories=use_columns_categories,
                               epochs=5,verbosity_level=3)


# train the model
hi.fit(df_train)


# In[7]:


dr_train_r = hi.transform(df_train)


# In[8]:


df_train.head()


# In[9]:


dr_train_r.head()


# # Within a pipeline

# In[10]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


# In[11]:


# create pre-processing data transformers
# uses the dictionary : 
# use_columns_categories = {'c0': 'real', 'c1': 'real', 'c2': 'real', 'c3': 'real'}
# and
# cols_use - from above


def create_transformers(cols_use,cols_target,cols_use_categories):
    # work on backup dataframe
    #df_ana_return = df_ana.copy()
    
    # count features (currently treated as numerical - using standard scaler
    count_features = [x for x in cols_use if cols_use_categories[x] in ['count'] and x not in cols_target]
    count_transformer = StandardScaler()


    # numeric features - assuming normal distribution
    # here I put in 'pos' - this would require anaother transformer on its own
    numeric_features = [x for x in cols_use if cols_use_categories[x] in ['real','num','pos'] and x not in cols_target]
    numeric_transformer = StandardScaler()

    # categorical features - use OHE (unkown ignored)
    categorical_features = [x for x in cols_use if cols_use_categories[x] in ['cat'] and x not in cols_target]
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # ordinal features - use OHE (unkown ignored)
    ordinal_features = [x for x in cols_use if cols_use_categories[x] in ['ord'] and x not in cols_target]
    for ord_att in ordinal_features:
        ord_att_num = '{}_org'.format(ord_att)
        if '{}_org'.format(ord_att) not in df_ana_return.columns:
            df_ana_return['{}_org'.format(ord_att)] = df_ana_return[ord_att].copy()
        #X[ord_att] = X['{}_org'.format(ord_att)].apply(lambda x: isinstance(x,str) and float(x.strip().split(' ')[0]) or x)
        df_ana_return[ord_att] = df_ana_return['{}_org'.format(ord_att)].apply(lambda x: float( (isinstance(x,str) and x.strip().split(' ')[0]) or np.nan ))

    ordinal_transformer = OrdinalEncoder(handle_unknown='ignore')


    preprocessor = ColumnTransformer(
        transformers=[
            ('count', count_transformer, count_features),
            ('real', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('ord', ordinal_transformer, ordinal_features)
        ],)
    return preprocessor, count_features, numeric_features, categorical_features , ordinal_features


preprocessor,count_features, numeric_features, categorical_features,ordinal_features = create_transformers(use_columns,target_columns,use_columns_categories)
preprocessor


# In[ ]:





# In[12]:


from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

cv_object = StratifiedKFold(n_splits=3,random_state=0,shuffle=True)


# In[13]:


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin

from tempfile import mkdtemp
# create temp dir
cachedir = mkdtemp()


pipes = {
    'DT':{
        'clf':Pipeline(
            steps=[
                ('imputer', hivae.impute.HiVAEImputer(
                    use_columns=use_columns,
                    impute_columns = use_columns,
                    target_columns=target_columns,
                    use_columns_categories=use_columns_categories,
                    epochs=5,
                    verbosity_level=1)),
                ('preprocessor', preprocessor),
                ('classifier', DecisionTreeClassifier())
                ],
#        ),
            memory = cachedir),
        'param_grid':{
            'classifier__criterion':                ['gini','entropy'],
            'classifier__max_depth':                [2,3],   #[2,3,4,5,6,7,8],
            'classifier__min_samples_split':        [3,],   #[2,3,4,5],
        }
    },    
}

def run_grid_search(X, y, classifier_title,clf,param_grid,cv_object,cv_inner=5,scoring='roc_auc',):

    # ,classifier__scoring=scoring)
    grid_search = GridSearchCV(clf, param_grid, cv=cv_inner)

    
    for i, (train, test) in enumerate(cv_object.split(X, y)):

        print('Fold : {}'.format(i))
        #print('run_grid_search',X.shape)
        # prepare for CV
        
        X_local_train = X.iloc[train].copy()
        y_local_train = y.iloc[train].copy()

        X_local_test = X.iloc[test].copy()
        y_local_test = y.iloc[test].copy()

        grid_search.fit(X_local_train, y_local_train),#,error_score='raise')
        print('Best params : {}'.format(grid_search.best_params_))

        # use best clasifier from internal grid
        classifier = grid_search.best_estimator_
        print(classifier)
    
        classifier.fit(X_local_train, y_local_train)
        
        used_colums = count_features+numeric_features
        if categorical_features:
            used_colums += list(classifier.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names(input_features=categorical_features))

        return grid_search    
            
X = df_train[use_columns] 
#display(X.head())
y = df_train[target_columns[0]]   # take the first feature in target_columns
#display(y.head())
  
    
classifier_choice = 'DT'        

start_time = time.time()

gs = run_grid_search(X, 
                y,
                classifier_choice,
                pipes[classifier_choice]['clf'],
                pipes[classifier_choice]['param_grid'],
                cv_object,
                cv_inner=3,
                scoring='roc_auc')

# remove temp dir
shutil.rmtree(cachedir)


end_time   = time.time()

time_elapsed = end_time - start_time

print(f'{time_elapsed} seconds')


# In[ ]:




