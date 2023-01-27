# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 20:07:44 2023

@author: sjilla
"""

import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
#from sklearn.externals import joblib #For exporting and importing
import joblib

#returns current working directory
os.getcwd()
#changes working directory
os.chdir('C://Users//sjilla//Downloads//titanic')

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()


X_titanic_train = titanic_train[['Pclass', 'SibSp', 'Parch']] #X-Axis
y_titanic_train = titanic_train['Survived'] #Y-Axis

#Build the decision tree model
dt = tree.DecisionTreeClassifier()
dt.fit(X_titanic_train, y_titanic_train)


#use cross validation to estimate performance of model. 
#==============================================================================
# cv_scores = model_selection. (dt, X_train, y_train, cv=5, verbose=3)
# cv_scores.mean()
#==============================================================================

#build final model on entire train data which is us for prediction
#dt.fit(X_train,y_train)

# natively deploy decision tree model(pickle format)
os.getcwd()
joblib.dump(dt, "Titanic.pkl")
