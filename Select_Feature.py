### Import library
import glob
import pandas as pd

### Set feature name of 12 feature set 
file_name = ["AtomPairs2D","AtomPairs2DCount","EState", "Extended", "Fingerprinterd", "GraphOnly",
"KlekotaRoth", "KlekotaRothCount", "MACCS", "Pubchem", "Substructure", "SubstructureCount"]
file_name = sorted(file_name) # Sorting

### Load origin test file in origin test folder
def test_folder():
    test_path = "Data/DILI_data/DILI_test_MF/*.csv"
    pathtest = []
    test_data = []
    for fname in glob.glob(test_path):
        pathtest.append(fname)
    pathtest = sorted(pathtest)   
    for path in pathtest:
        df = pd.read_csv(path)
        test_data.append(df)
    return test_data

### Load train file in train folder after feature selecting
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

### Select features colume in test base on train dataset after feature selecting
### Also adding test label 
test_label = pd.read_csv("Data/DILI_data/DILI_test_MF/DILI_test_AtomPairs2D.csv") #Load test file contain test labels
train_data = train_folder()
test_data = test_folder()
for train, test, name in zip(train_data, test_data, file_name):
    test_select = test.loc[:,train.columns]
    test_select["class."] = test_label["class."]
    test_select.isnull().values.any()
    test_path = "Data/Feature_Data/Feature_Test/Feature_Test_"+ name +".csv"
    test_select.to_csv(test_path, index = None,header = True)
