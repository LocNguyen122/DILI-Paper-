### Import library 
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef, accuracy_score, f1_score
import os
import glob
from sklearn.model_selection import train_test_split

### import model was selected as classifier 
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

### Evaluation
def score_model(y_test,y_predicted):
    cm = confusion_matrix(y_test, y_predicted)
    accuracy.append(accuracy_score(y_test, y_predicted))
    roc_auc.append(roc_auc_score(y_test, y_predicted))
    f1.append(f1_score(y_test, y_predicted))
    sensitivity.append(cm[0,0] / (cm[0,0]+cm[0,1]))
    specificity.append(cm[1,1] / (cm[1,0]+cm[1,1]))
    MCC.append(matthews_corrcoef(y_test, y_predicted))

### Define model with parameter 
def Model_Seed(seed):
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state= seed)
    XGB_Fingerprinterd = xgb.XGBClassifier(n_estimators=200,random_state=seed,n_jobs=-1,max_depth=5,max_features=0.2)
    XGB_KlekotaRothCount = xgb.XGBClassifier(n_estimators=200,random_state=seed,n_jobs=-1,max_depth=6,max_features=0.2)
    SVM_KlekotaRoth = svm.SVC(random_state=seed, probability= True,C=5,gamma=5)
    SVM_Pubchem = svm.SVC(random_state=seed, probability= True,C=1,gamma=5)
    KNN_Extended = KNeighborsClassifier(n_jobs=-1,n_neighbors=10,weights="distance")
    KNN_MACCS = KNeighborsClassifier(n_jobs=-1,n_neighbors=9,weights="distance")
    return skf, XGB_Fingerprinterd, XGB_KlekotaRothCount, SVM_KlekotaRoth, SVM_Pubchem, KNN_Extended, KNN_MACCS

### Tune definition
def tune_model(train, test, model, CV):
    y_trainval = train["class."]
    X_trainval = train.drop(labels = "class.", axis = 1)
    y_test = test["class."]
    X_test= test.drop(labels = "class.", axis = 1)
    X_trainval = Normalizer().fit_transform(X_trainval)
    X_test = Normalizer().transform(X_test)
    model.fit(X_trainval, y_trainval)
    y_predicted = model.predict(X_test)
    y_proba_trainval = list(model.predict_proba(X_trainval)[:,1])
    y_proba_test = list(model.predict_proba(X_test)[:,1])
    return y_test, y_predicted, y_proba_test, y_proba_trainval, y_trainval, y_test

### Create list and dataframe for storage 
SEED = []
accuracy = []
roc_auc = []
f1 = []
sensitivity =[]
specificity = []
MCC = []
Test_proba = pd.DataFrame()
Train_proba = pd.DataFrame()

### Seed range
total_seed = range(1,11)

### Set list for loop
model_name = ["_RF", "_XGB", "_SVM", "_KNN", "_LR","_MLP"]

file_name = ["XGB_Fingerprinterd","XGB_KlekotaRothCount","SVM_KlekotaRoth", "SVM_Pubchem", "KNN_Extended", "KNN_MACCS"]

train_path =["Data/Feature_Data/Feature_Train/Train_FeatureSelector_Fingerprinterd.csv",
            "Data/Feature_Data/Feature_Train/Train_FeatureSelector_KlekotaRothCount.csv",
            "Data/Feature_Data/Feature_Train/Train_FeatureSelector_KlekotaRoth.csv",
            "Data/Feature_Data/Feature_Train/Train_FeatureSelector_Pubchem.csv",
            "Data/Feature_Data/Feature_Train/Train_FeatureSelector_Extended.csv",
            "Data/Feature_Data/Feature_Train/Train_FeatureSelector_MACCS.csv"]

test_path =["Data/Feature_Data/Feature_Test/Feature_Test_Fingerprinterd.csv",
            "Data/Feature_Data/Feature_Test/Feature_Test_KlekotaRothCount.csv",
            "Data/Feature_Data/Feature_Test/Feature_Test_KlekotaRoth.csv",
            "Data/Feature_Data/Feature_Test/Feature_Test_Pubchem.csv",
            "Data/Feature_Data/Feature_Test/Feature_Test_Extended.csv",
            "Data/Feature_Data/Feature_Test/Feature_Test_MACCS.csv"]

### Train model
for seed in total_seed:
    Model_outcome = Model_Seed(seed)
    CV = Model_outcome[0]
    XGB_Fingerprinterd, XGB_KlekotaRothCount, SVM_KlekotaRoth = Model_outcome[1], Model_outcome[2], Model_outcome[3]
    SVM_Pubchem, KNN_Extended, KNN_MACCS = Model_outcome[4], Model_outcome[5], Model_outcome[6]
    total_model = [XGB_Fingerprinterd, XGB_KlekotaRothCount, SVM_KlekotaRoth, SVM_Pubchem, KNN_Extended, KNN_MACCS]
    for model, name, train, test in zip(total_model, file_name, train_path, test_path):
        print(seed, model)
        train_data, test_data = pd.read_csv(train), pd.read_csv(test)
        tune_model_result = tune_model(train_data, test_data, model, CV)
        y_test = tune_model_result[0]
        y_predicted = tune_model_result[1]
        score_model(y_test,y_predicted)
        Test_proba[name] = tune_model_result[2]
        Train_proba[name] = tune_model_result[3]
        SEED.append(seed)
    df = pd.DataFrame({"Seed":seed,"Accuracy":accuracy,"ROC_AUC_score":roc_auc ,"F1_score":f1,"Sensitivity":sensitivity,"Specificity":specificity,"MCC":MCC},index= file_name)
    df.to_csv("6_Classifier2_"+"{0:03}".format(seed)+".csv",header=True)
    Test_proba["class."] = tune_model_result[5]
    Test_proba.to_csv("6_Classifer2_Test_Predict_"+"{0:03}".format(seed)+".csv", header= True,index = None)
    Train_proba["class."] = tune_model_result[4]
    Train_proba.to_csv("6_Classifer2_Train_Predict_"+"{0:03}".format(seed)+".csv", header= True, index =None)
    SEED = []
    accuracy = []
    roc_auc = []
    f1 = []
    sensitivity =[]
    specificity = []
    MCC = []
    Test_proba = pd.DataFrame()
    Train_proba = pd.DataFrame()



    
