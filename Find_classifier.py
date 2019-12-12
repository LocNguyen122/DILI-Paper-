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

### Set StratifiedKFold and scaler 
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state= 0)

### Extreme Gradient Boosting 
import xgboost as xgb
max_depth = np.arange(1, 7, 1)
max_features = np.arange(0.2, 0.93, 0.05)
XGB_parameters = {
    "max_depth": max_depth,
    "max_features": max_features
}
 
### Random forest 
max_depth = np.arange(1, 7, 1)
max_features = np.arange(0.2, 0.93, 0.05)
RF_parameters = {
    "max_depth": max_depth,
    "max_features": max_features
}

### SVM
from sklearn import svm
Cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
SVM_parameters = {"C": Cs, "gamma" : gammas}

### KNN
from sklearn.neighbors import KNeighborsClassifier
k_range = list(range(3,11))
weight_options = ["uniform", "distance"]
KNN_parameters = dict(n_neighbors = k_range, weights = weight_options)

### LR 
from sklearn.linear_model import LogisticRegression
C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
LR_parameters = {"C": C}

### MLP
from sklearn.neural_network import MLPClassifier
MLP_parameters = {"hidden_layer_sizes": [(64,128,64), (64,128,256), (256,128,64),
                (64,128,256,64), (64,128,256,512), (512,256,128,64),
                (64,128,256,512,64), (64,128,256,512,1024), (1024,512,256,128,64)]}

### Evaluation
def score_model(y_test,y_predicted):
    cm = confusion_matrix(y_test, y_predicted)
    accuracy.append(accuracy_score(y_test, y_predicted))
    roc_auc.append(roc_auc_score(y_test, y_predicted))
    f1.append(f1_score(y_test, y_predicted))
    sensitivity.append(cm[0,0] / (cm[0,0]+cm[0,1]))
    specificity.append(cm[1,1] / (cm[1,0]+cm[1,1]))
    MCC.append(matthews_corrcoef(y_test, y_predicted))

### Define model
def Model_Seed(seed):
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state= seed)
    RF = RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=-1, class_weight="balanced")
    XGB = xgb.XGBClassifier(n_estimators=200,random_state=seed,n_jobs=-1)
    SVM = svm.SVC(random_state=seed, probability= True)
    KNN = KNeighborsClassifier(n_jobs=-1)
    LR = LogisticRegression(random_state=seed, n_jobs=-1)
    MLP = MLPClassifier(random_state=seed)
    return skf, RF, XGB, SVM, KNN, LR, MLP

### Tune model
def tune_model(train, test, model, parameters, CV):
    y_trainval = train["class."]
    X_trainval = train.drop(labels = "class.", axis = 1)
    y_test = test["class."]
    X_test= test.drop(labels = "class.", axis = 1)
    X_trainval = Normalizer().fit_transform(X_trainval)
    X_test = Normalizer().transform(X_test)
    grid = GridSearchCV(estimator= model, param_grid= parameters,cv=CV.get_n_splits(X_trainval, y_trainval), n_jobs=-1,scoring = "roc_auc",return_train_score = True, iid = False)
    grid.fit(X_trainval, y_trainval)
    best_params = grid.best_params_
    best_model = grid.best_estimator_
    grid_score.append(grid.best_score_) #test score
    best_model.fit(X_trainval, y_trainval)
    y_predicted = best_model.predict(X_test)
    y_proba_trainval = list(best_model.predict_proba(X_trainval)[:,1])
    y_proba_test = list(best_model.predict_proba(X_test)[:,1])
    result = grid.cv_results_
    return y_test,y_predicted, result, best_params, y_proba_test, y_proba_trainval

### Load test file in test folder
def test_folder():
    test_path = "Data/Feature_Data/Feature_Test/*.csv"
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
    train_path = "Data/Feature_Data/Feature_Train/*.csv"
    pathtrain = []
    train_data = []
    for fname in glob.glob(train_path):
        pathtrain.append(fname)
    pathtrain = sorted(pathtrain)   
    for path in pathtrain:
        df = pd.read_csv(path)
        train_data.append(df)
    return train_data

### Create list
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

### Seed range
total_seed = range(1,5)

### Load dataset
train_dataset = train_folder()
test_dataset = test_folder()

### Set list for loop
total_para = [RF_parameters, XGB_parameters, SVM_parameters, KNN_parameters, LR_parameters, MLP_parameters]
model_name = ["_RF", "_XGB", "_SVM", "_KNN", "_LR","_MLP"]

file_name = ["AtomPairs2D","AtomPairs2DCount","EState", "Extended", "Fingerprinterd", "GraphOnly",
"KlekotaRoth", "KlekotaRothCount", "MACCS", "Pubchem", "Substructure", "SubstructureCount"]

### Train model
for seed in total_seed:
    Model_outcome = Model_Seed(seed)
    CV = Model_outcome[0]
    RF, XGB, SVM = Model_outcome[1], Model_outcome[2], Model_outcome[3]
    KNN, LR, MLP = Model_outcome[4], Model_outcome[5], Model_outcome[6]
    total_model = [RF, XGB, SVM, KNN, LR, MLP]
    for model, parameter, name_model in zip(total_model, total_para, model_name):
        print(name_model, seed, model, CV)
        for name, train, test in zip(file_name, train_dataset, test_dataset):
            tune_model_result = tune_model(train,test, model, parameter, CV)
            y_test = tune_model_result[0]
            y_predicted = tune_model_result[1]
            best_params.append(tune_model_result[3])
            score_model(y_test,y_predicted)
            Test_proba[name] = tune_model_result[4]
            Train_proba[name] = tune_model_result[5]
        df = pd.DataFrame({"5_Fold CV":grid_score,"Accuracy":accuracy,"ROC_AUC_score":roc_auc ,"F1_score":f1,"Sensitivity":sensitivity,"Specificity":specificity,"MCC":MCC, "best_params":best_params},index= file_name)
        df.to_csv("Result/"+name_model[1:]+"/Test_Score/"+str(seed) + name_model + "_Test_Score_.csv",header=True)
        Test_proba.to_csv("Result/"+name_model[1:]+"/Test_Proba/"+str(seed) + name_model + "_Test_Proba.csv", header= True,index = None)
        Train_proba.to_csv("Result/"+name_model[1:]+"/Train_Proba/"+str(seed) + name_model + "_Train_Proba.csv", header= True, index =None)
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



    
