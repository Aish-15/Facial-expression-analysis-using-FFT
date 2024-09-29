# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 21:06:38 2021

@author: aishg
"""

import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib

happy = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/Happyface.csv')
sad = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/SadExpressionsP.csv')
disgusted = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/DisgExpressionsP.csv')
surprised = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/Surpface.csv')
fear = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fearface.csv')

framesAll = [happy, sad, disgusted, surprised, fear]
framestogether = pd.concat(framesAll)
indx1 = np.arange(len(framestogether))
AllExp = np.random.permutation(indx1)
AllExp = framestogether.sample(frac=1).reset_index(drop=True)
AllExp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/AllExpMerged.csv', index=False)

framestogether.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/AllExpJustMerged.csv', index=False)

Exp_train, Exp_test = train_test_split(AllExp, test_size=0.2, random_state = 44)
Exp_train.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/TrainExp.csv', index=False)
Exp_test.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/TestExp.csv', index=False)

train = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/TrainExp.csv')
test = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/TestExp.csv')

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense


y = AllExp['65']
AllExp.drop('65',axis=1,inplace=True)
X = AllExp

tmp = np.array(X)
X1 = tmp[:,0:65]

Y1 = np.array(y)

row, col = X.shape

TR = round(row*0.8)# Training with 80% data
X1_train = X1[0:TR-1,:]
Y1_train = Y1[0:TR-1]

from sklearn import svm

obj = svm.SVC()
obj.fit(X1_train,Y1_train)


X1_test = X1[TR:row,:]
y_test = y[TR:row]

yhat_test = obj.predict(X1_test)



from sklearn import metrics
from sklearn.metrics import classification_report

print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, yhat_test))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, yhat_test, average = 'weighted'))
print('\nClassification Report of MCR dataset\n')
print(classification_report(y_test, yhat_test, target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']))

clf = svm.SVC(kernel='linear', C=1).fit(X1_train, Y1_train)
clf.score(X1_test, y_test)

#Overall accuracy score is 25.538

from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)

## Individual clf scores is found to be [0.23384615, 0.24923077, 0.21846154, 0.22461538, 0.26234568]

from sklearn import metrics
scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
