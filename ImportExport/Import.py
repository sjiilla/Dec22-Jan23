# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 20:08:05 2023

@author: sjilla
"""

import os
import pandas as pd
#from sklearn.externals import joblib
import joblib

#changes working directory
os.chdir('C://Users//sjilla//Downloads//titanic')


titanic_test = pd.read_csv("test.csv")
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()
X_titanic_test = titanic_test[['Pclass', 'SibSp', 'Parch']] #X-Axis
#y_titanic_train = titanic_test['Survived'] #Y-Axis

os.getcwd()

#No prediction logic here

#Use load method to load Pickle file
serv2 = joblib.load("Titanic.pkl")
titanic_test['Survived'] = serv2.predict(X_titanic_test)
titanic_test.to_csv("submissionUsingJobLib2.csv", columns=['PassengerId','Survived'], index=False)
