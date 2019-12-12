### Import library 
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

### Set StratifiedKFold and scaler 
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)

### Extreme Gradient Boosting 
import xgboost as xgb
max_depth = np.arange(1, 7, 1)
max_features = np.arange(0.2, 0.93, 0.05)
XGB_parameters = {
    'max_depth': max_depth
}
XGB = xgb.XGBClassifier(n_estimators=200,random_state=0,n_jobs=-1)

### Random forest 
max_depth = np.arange(1, 7, 1)
max_features = np.arange(0.2, 0.93, 0.05)
RF_parameters = {
    'max_depth': max_depth
}
RF = RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=-1, class_weight="balanced")

### SVM
from sklearn import svm
Cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
SVM_parameters = {'C': Cs, 'gamma' : gammas}
SVM = svm.SVC(random_state=0, probability= True)

### KNN
from sklearn.neighbors import KNeighborsClassifier
k_range = list(range(3,11))
weight_options = ["uniform", "distance"]
KNN_parameters = dict(n_neighbors = k_range, weights = weight_options)
KNN = KNeighborsClassifier(n_jobs=-1)

### LR 
from sklearn.linear_model import LogisticRegression
C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
LR_parameters = {'C': C}
LR = LogisticRegression(random_state=0, n_jobs=-1)

### MLP
from sklearn.neural_network import MLPClassifier
MLP_parameters = {'hidden_layer_sizes': [(64,128,64), (64,128,256), (256,128,64),
                (64,128,256,64), (64,128,256,512), (512,256,128,64),
                (64,128,256,512,64), (64,128,256,512,1024), (1024,512,256,128,64)]}
MLP = MLPClassifier(random_state=0)

### Evaluation  
def score_model(y_test,y_predicted):
    cm = confusion_matrix(y_test, y_predicted)
    accuracy.append(accuracy_score(y_test, y_predicted))
    roc_auc.append(roc_auc_score(y_test, y_predicted))
    f1.append(f1_score(y_test, y_predicted))
    sensitivity.append(cm[0,0] / (cm[0,0]+cm[0,1]))
    specificity.append(cm[1,1] / (cm[1,0]+cm[1,1]))
    MCC.append(matthews_corrcoef(y_test, y_predicted))

### Tune definition
def tune_model(train, test, model, parameters):
    y_trainval = train["class."]
    X_trainval = train.drop(labels = "class.", axis = 1)
    y_test = test["class."]
    X_test= test.drop(labels = "class.", axis = 1)
    X_trainval = Normalizer().fit_transform(X_trainval)
    X_test = Normalizer().transform(X_test)
    rf_grid = GridSearchCV(estimator= model, param_grid= parameters,cv=skf.get_n_splits(X_trainval, y_trainval), n_jobs=-1,scoring = 'roc_auc',return_train_score = True, iid = False)
    rf_grid.fit(X_trainval, y_trainval)
    best_params = rf_grid.best_params_
    best_model = rf_grid.best_estimator_
    grid_score.append(rf_grid.best_score_) #test score
    best_model.fit(X_trainval, y_trainval)
    y_predicted = best_model.predict(X_test)
    y_proba_trainval = list(best_model.predict_proba(X_trainval)[:,1])
    y_proba_test = list(best_model.predict_proba(X_test)[:,1])
    result = rf_grid.cv_results_
    return y_test,y_predicted, result, best_params, y_proba_test, y_proba_trainval

### Define list for loop
total_model = [SVM, RF, XGB, KNN, MLP, LR]
total_para = [SVM_parameters, RF_parameters, XGB_parameters, KNN_parameters, MLP_parameters, LR_parameters]
model_name = ["_SVM","_RF", "_XGB",  "_KNN", "_MLP","_LR"]
file_name = ["XGB_Fingerprinterd","XGB_KlekotaRothCount","SVM_KlekotaRoth", "SVM_Pubchem", "KNN_Extended", "KNN_MACCS"]

### Load test file in test folder
def test_folder():
    test_path = "Test_6_Classifier/Train_Proba/*.csv"
    pathtest = []
    test_data = []
    for fname in glob.glob(test_path):
        pathtest.append(fname)
    pathtest = sorted(pathtest)   
    for path in pathtest:
        df = pd.read_csv(path)
        test_data.append(df)
    return test_data

### Load train file in train folder
def train_folder():
    train_path = "Test_6_Classifier/Test_Proba/*.csv"
    pathtrain = []
    train_data = []
    for fname in glob.glob(train_path):
        pathtrain.append(fname)
    pathtrain = sorted(pathtrain)   
    for path in pathtrain:
        df = pd.read_csv(path)
        train_data.append(df)
    return train_data

### Create list and dataframe for storage 
grid_score = []
accuracy = []
roc_auc = []
f1 = []
sensitivity =[]
specificity = []
MCC = []
best_params =[]
n_seed = []
n_model =[]
Test_proba = pd.DataFrame()
Train_proba = pd.DataFrame()

### Set range of seed
total_seed = range(1,11) #31

### Tuning model
for seed in total_seed:
    train_set = train_folder()
    test_set = test_folder()
    train, test = train_set[seed-1],test_set[seed-1]
    for model, parameter, name_model in zip(total_model, total_para, model_name):
        print(seed, name_model)
        n_seed.append(seed)
        n_model.append(name_model[1:])
        tune_model_result = tune_model(train,test, model, parameter)
        y_test = tune_model_result[0]
        y_predicted = tune_model_result[1]
        best_params.append(tune_model_result[3])
        score_model(y_test,y_predicted)
        Test_proba = pd.DataFrame(tune_model_result[4])
        Train_proba = pd.DataFrame(tune_model_result[5])
        df = pd.DataFrame({'Seed': n_seed,'Model':n_model,'Grid_score':grid_score,'Accuracy':accuracy,'ROC_AUC_score':roc_auc ,'F1_score':f1,'Sensitivity':sensitivity,'Specificity':specificity,'MCC':MCC, 'best_params':best_params},index= None)
        df.to_csv('Ensemble_Result_Test/'+name_model[1:]+'/Test_Score/' + name_model[1:] + '_Ensemble_Test_Score_'+"{0:03}".format(seed)+'.csv',header=True)
        Test_proba.to_csv('Ensemble_Result_Test/'+name_model[1:]+'/Test_Proba/' + name_model[1:] + '_Ensemble_Test_Proba_'+"{0:03}".format(seed)+'.csv', header= True,index = None)
        Train_proba.to_csv('Ensemble_Result_Test/'+name_model[1:]+'/Train_Proba/' + name_model[1:] + '_Ensemble_Train_Proba_'+"{0:03}".format(seed)+'.csv', header= True, index = None)
        grid_score = []
        accuracy = []
        roc_auc = []
        f1 = []
        sensitivity =[]
        specificity = []
        MCC = []
        best_params =[]
        Test_proba = pd.DataFrame()
        Train_proba = pd.DataFrame()
        n_seed = []
        n_model =[]
            

