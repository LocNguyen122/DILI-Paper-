# Import library 
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef, accuracy_score, f1_score
import os
import glob

# Evaluation 
def score_model(y_test,y_predicted):
    cm = confusion_matrix(y_test, y_predicted)
    accuracy.append(accuracy_score(y_test, y_predicted))
    roc_auc.append(roc_auc_score(y_test, y_predicted))
    f1.append(f1_score(y_test, y_predicted))
    sensitivity.append(cm[0,0] / (cm[0,0]+cm[0,1]))
    specificity.append(cm[1,1] / (cm[1,0]+cm[1,1]))
    MCC.append(matthews_corrcoef(y_test, y_predicted))

#Tune definition
def tune_model(train, test):
    y_trainval = train["class."]
    X_trainval = train.drop(labels = "class.", axis = 1)
    y_test = test["class."]
    X_test= test.drop(labels = "class.", axis = 1)
    X_trainval["mean"] = X_trainval.mean(axis=1)
    X_test["mean"] = X_test.mean(axis=1)
    y_proba_trainval = X_trainval["mean"]
    y_proba_test = X_test["mean"]
    y_predicted = X_test["mean"].round()
    return y_test,y_predicted, y_proba_test, y_proba_trainval

# File was selected for classifier
file_name = ["XGB_Fingerprinterd","XGB_KlekotaRoth","SVM_KlekotaRoth", "SVM_Pubchem", "KNN_Extended", "KNN_MACCS"]

# Load test file in test folder
def test_folder():
    test_path = "6 Classifier/Test_Proba/*.csv"
    pathtest = []
    test_data = []
    for fname in glob.glob(test_path):
        pathtest.append(fname)
    pathtest = sorted(pathtest)   
    for path in pathtest:
        df = pd.read_csv(path)
        test_data.append(df)
    return test_data

# Load train file in train folder
def train_folder():
    train_path = "6 Classifier/Test_Proba/*.csv"
    pathtrain = []
    train_data = []
    for fname in glob.glob(train_path):
        pathtrain.append(fname)
    pathtrain = sorted(pathtrain)   
    for path in pathtrain:
        df = pd.read_csv(path)
        train_data.append(df)
    return train_data

#Create list for storage
accuracy = []
roc_auc = []
f1 = []
sensitivity =[]
specificity = []
MCC = []
best_params =[]
n_seed = []
Test_output = pd.DataFrame()
Train_output = pd.DataFrame()

# Range of seed 
total_seed = range(1,11) #31

# Tuning model
for seed in total_seed:
    train_set = train_folder()
    test_set = test_folder()
    train, test = train_set[seed-1],test_set[seed-1]
    print(seed)
    n_seed.append(seed)
    tune_model_result = tune_model(train,test)
    y_test = tune_model_result[0]
    y_predicted = tune_model_result[1]
    score_model(y_test,y_predicted)
    Test_output.append(tune_model_result[2])
    Train_output.append(tune_model_result[3])
    df = pd.DataFrame({'Seed': n_seed,'Accuracy':accuracy,'ROC_AUC_score':roc_auc ,'F1_score':f1,'Sensitivity':sensitivity,'Specificity':specificity,'MCC':MCC},index= None)
    df.to_csv('Average_Ensemble_Test_Score_'+"{0:03}".format(seed)+'.csv',header=True)
    Test_output.to_csv('Average_Ensemble_Test_Predict'+"{0:03}".format(seed)+'.csv', header= True,index = None)
    Train_output.to_csv('Average_Ensemble_Train_Predict'+"{0:03}".format(seed)+'.csv', header= True,index = None)
    accuracy = []
    roc_auc = []
    f1 = []
    sensitivity =[]
    specificity = []
    MCC = []
    best_params =[]
    n_seed=[]
    Test_output = pd.DataFrame()
    Train_output = pd.DataFrame()
            
