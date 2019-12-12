# Import library
import pandas as pd
import numpy as np
import os
import glob
from feature_selector import FeatureSelector

# Set path of origin training data
train_path = "Data/DILI_data/DILI_train_MF/*.csv"

# Create list for storage
pathtrain = []
train_data = []

# Load path in training folder
for fname in glob.glob(train_path):
    pathtrain.append(fname)
pathtrain = sorted(pathtrain)   

# Load csv file as dataframe from path
for path in pathtrain:
    df = pd.read_csv(path, low_memory=False)
    train_data.append(df)

# Define name of 12 features set 
file_name = ["AtomPairs2D","AtomPairs2DCount","EState", "Extended", "Fingerprinterd", "GraphOnly",
"KlekotaRoth", "KlekotaRothCount", "MACCS", "Pubchem", "Substructure", "SubstructureCount"]
file_name = sorted(file_name) # Sorting name

#################
#Load one train data for get labels
train_label = pd.read_csv("Data/DILI_data/DILI_train_MF/DILI_train_AtomPairs2D.csv")

# Start feature selecting and add labels for each training dataset
for train, name in zip(train_data, file_name):
    feature_columns = []
    labels = train_label["class."]
    X_train = train.drop(labels = "Name", axis = 1)
    fs = FeatureSelector(data = X_train, labels = labels)
    fs.identify_all(selection_params = {'missing_threshold': 0.8, 'correlation_threshold': 0.98, 
                                        'task': 'classification', 'eval_metric': 'auc', 
                                        'cumulative_importance': 0.99,'num_threads':-1})
    train_removed_all = fs.remove(methods = 'all', keep_one_hot=False) 
    print('Original Number of Features', train.shape[1]) 
    print('Final Number of Features: ', train_removed_all.shape[1]) 
    train_removed_all.head()
    feature_columns.extend(train_removed_all.columns)
    feature_columns = pd.DataFrame(feature_columns,index=None)
    feature_columns.to_csv('Features_'+ name+'.csv',index = False, header = name)
    train_removed_all['class.']=labels
    train_removed_all.to_csv('Data/Feature_Data/Feature_Data/Feature_Train_'+ name + '.csv', index=False, header=True)

