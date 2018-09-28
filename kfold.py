
####################################################
# PCA function for performing kfold cross validation
# Author : Devin Upreti 
####################################################


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


####################################################
# perfoms kfold validation on provided data
# assumes target variable is in the column 'label'
def kfold(data):
    X = reduced_dimension_data
    X = X.sample(frac=1).reset_index(drop=True)
    y = X.label
    kf = KFold(n_splits=5)
    
    test_number = 0
    fscore_average = [0, 0]
    print("")
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        logisticRegr = LogisticRegression()
        logisticRegr.fit(X_train, y_train)

        predictions = logisticRegr.predict(X_test)
        test_number += 1
        print("TEST ",test_number)
        print("Confusion Matrix")
        print("-------------------")
        cm = metrics.confusion_matrix(y_test, predictions)
        print(cm)
        print("-------------------\n")

        precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, predictions)
        print('precision: ',precision)
        print('recall:    ', recall)
        print('fscore:    ', fscore)
        print("====================================\n")
        fscore_average += fscore
        
    fscore_average = fscore_average / test_number
    print("Average Fscore over kFold: ", fscore_average)
    print("====================================\n")      

