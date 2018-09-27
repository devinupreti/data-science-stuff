
####################################################
# Function for logistic regression and testing
# Author : Devin Upreti 
####################################################


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

###################################################
# Assumes target class is 'label'
def trainandtest(data):
    # Shuffle
    data = data.sample(frac=1).reset_index(drop=True)

    y = data.label
    X = data
    # 70-30 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)

    predictions = logisticRegr.predict(X_test)
    
    #############################################################
    # Useful for imbalanced classes
    #############################################################
    print("TEST ON SAMPLED DATA")
    print("Confusion Matrix")
    print("-------------------")
    cm = metrics.confusion_matrix(y_test, predictions)
    print(cm)
    print("-------------------\n")

    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, predictions)
    print('precision: ',precision)
    print('recall:    ', recall)
    print('fscore:    ', fscore)
    
    