import warnings
warnings.simplefilter(action='once', category=UserWarning) # issues with : UserWarning: `tf.layers.dense` is deprecated and will be removed in a future version. Please use `tf.keras.layers.Dense` instead.

import pandas as pd
import os,os.path
import hivae
import numpy as np
import pprint
printer = pprint.PrettyPrinter(depth=3).pprint
import scipy.stats


main_directory = './'

# select which data to run for this example
dataset_name = 'Diabetes'
#dataset_name = 'Adult'
#dataset_name = 'Mock'

# set up paths
# where the data should be found
dataset_path = '{}/data/{}'.format(main_directory,dataset_name)
# where the results will be saved
results_path = '{}/results/{}'.format(main_directory,dataset_name)
# where the networks will be saved
network_path = '{}/network/{}'.format(main_directory,dataset_name)


# generate information for different datasets
types_list_d = {} # empty hash 
#Diabetes
types_list_d['Diabetes'] = [
    ('AGE','pos',1,None), # positive numerical , use one feature
    ('SEX','cat',2,2),    # categorical - has two values , use two features
    ('BMI','pos',1,None),
    ('BP','pos',1,None),
    ('S1','pos',1,None),
    ('S2','pos',1,None),
    ('S3','pos',1,None),
    ('S4','pos',1,None),
    ('S5','pos',1,None),
    ('S6','pos',1,None),
    # ('Y','pos',1,None)
    ]

# Adult
types_list_d['Adult'] = [
    ('V1','count',1,None),  # count values, use one feature
    ('V2','cat',7,7),
    ('V3','pos',1,None),
    ('V4','ordinal',16,16),
    ('V5','cat',7,7),
    ('V6','cat',14,14),
    ('V7','cat',6,6),
    ('V8','cat',5,5),
    ('V9','cat',2,2),
    ('V10','pos',1,None),
    ('V11','pos',1,None),
    ('V12','count',1,None)  # count values, use one feature
    ]

types_list_d['Mock'] = [
    ('V1','count',1,None),
    ('V2','cat',2,2),
    ('V3','ordinal',2,2),
    ('V4','pos',1,None),
    ]
    


# use for the selected dataset 
types_list = types_list_d[dataset_name]

# assumes that in dataset_path (e.g ./data/Adult)
data_file_org = '{}/data_org.csv'.format(dataset_path)
data_file     = '{}/data.csv'.format(dataset_path) 
data_file_id  = '{}/data_id.csv'.format(dataset_path)
train_file    = '{}/data_train.csv'.format(dataset_path) # has to exits!!!!!
test_file     = '{}/data_test.csv'.format(dataset_path)

train_data = None # will contain the pandas dataframe for training
test_data  = None # will contain the pandas dataframe for testing

# if train/test does not exist - create from new
if not os.path.exists(train_file):
    data_df = pd.read_csv(data_file_org,header=None)
    data_df.insert(0,'ID',['{:s}'.format(str(x).zfill(3)) for x in range(len(data_df))])
    print('Len data = ',len(data_df))
    # save data including IDs 
    data_df.to_csv(data_file_id,header=False,index=False)
    # drop IDs again
    data_df = data_df.drop('ID', axis=1)
    data_df.to_csv(data_file,header=False,index=False)

    test_data = data_df.sample(frac=0.3,replace=False,random_state=33)
    train_data = data_df.loc[set(data_df.index)-set(test_data.index)]

    train_data.to_csv(train_file,header=False,index=False)
    test_data.to_csv(test_file,header=False,index=False)
#if not os.path.exists(train_file):
else:
    train_data = pd.read_csv(train_file)
    test_data  = pd.read_csv(test_file)


# construct missing data mask - for data having missing data
missing_true_train = pd.DataFrame()
for x in list(train_data.columns.values):
    missing_true_train[x] = train_data[x].isna().map({True:0,False:1})

missing_true_test = pd.DataFrame()
for x in list(test_data.columns.values):
    missing_true_test  =  test_data[x].isna().map({True:0,False:1})
    
    
# print(len(train_data))
# print(list(train_data.index[:20]))
# print(len(test_data))
# print(list(test_data.index[:20]))

network_dict = {
    'Diabetes':{
        'batch_size' : 32,
        'model_name': 'model_HIVAE_inputDropout',
        'dim_z': 5,
        'dim_y': 5,
        'dim_s': 10,
    },
    'Adult':{
        'batch_size' : 32,
        'model_name': 'model_HIVAE_inputDropout',
        'dim_z': 8,
        'dim_y': 10,
        'dim_s': 10,
    },
    'Mock':{
        'batch_size' : 32,
        'model_name': 'model_HIVAE_inputDropout',
        'dim_z': 3,
        'dim_y': 3,
        'dim_s': 3,
    }
}
iterations = {
    'Diabetes':50,
    'Adult':10,
    'Mock':10
}
printer(types_list)
printer('{}: {}'.format('len training data',len(train_data)))
printer(train_data)
printer(missing_true_train)
hivae_obj = hivae.hivae(types_list,network_dict[dataset_name],network_path,results_path)

hivae_obj.fit(train_data,epochs=iterations[dataset_name],true_missing_mask=missing_true_train)
(test_data, test_data_reconstructed, test_data_decoded, test_data_embedded_z, test_data_embedded_s) = hivae_obj.predict(test_data,true_missing_mask=missing_true_test)

df_test_data               = pd.DataFrame(test_data)
df_test_data_reconstructed = pd.DataFrame(test_data_reconstructed)
df_test_data_decoded       = pd.DataFrame(test_data_decoded)
df_test_data_embedded_z    = pd.DataFrame(test_data_embedded_z)
df_test_data_embedded_s    = pd.DataFrame(test_data_embedded_s)

printer(df_test_data_embedded_z)
printer('-'*80)
printer('df_test_data')
printer(df_test_data)
#printer('-'*80)
# printer('df_test_data_reconstructed')
# printer(df_test_data_reconstructed)
printer('-'*80)
printer('df_test_data_decoded')
printer(df_test_data_decoded)
printer('-'*80)
printer(pd.DataFrame(test_data-test_data_decoded))
printer('-'*80)
printer('Column-wise corrlations')

column_names = [x[0] for x in types_list_d[dataset_name]]
print('{:20}\t{:10}\t{:10}'.format('Column Name','Pearson','p value'))

for i in range(len(column_names)):
    corr,pval = scipy.stats.pearsonr(test_data[:,i],test_data_decoded[:,i])
    print('{:20}\t{:5.2f}\t{:f}'.format(column_names[i],corr,pval))
