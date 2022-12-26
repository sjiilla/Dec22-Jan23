# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 20:48:25 2022

@author: sjilla
"""

# -*- coding: utf-8 -*-
"""
@author: Sreeni Jilla
"""
import pandas as pd
#sklearn : Scintific Kit Learn/ SciKit Learn
from sklearn import tree #For Decissin Tree

#Read Train Data file
titanic_train = pd.read_csv("E:\\Data Science\\Data\\titanic_train.csv")
#os.chdir("E:/Data Science/Data/")

titanic_train.shape 
titanic_train.info() 
titanic_train.describe()

#Let's start the journey with non categorical and non missing data columns
X_titanic_train = titanic_train[['Pclass', 'SibSp', 'Parch']] #X-Axis
y_titanic_train = titanic_train['Survived'] #Y-Axis

#Build the decision tree model
dt = tree.DecisionTreeClassifier()
dt.fit(X_titanic_train, y_titanic_train)

#Predict the outcome using decision tree
#Read the Test Data
titanic_test = pd.read_csv("E:\\Data Science\\Data\\titanic_test.csv")
X_test = titanic_test[['Pclass', 'SibSp', 'Parch']]
#Use .predict method on Test data using the model which we built
titanic_test['Survived'] = dt.predict(X_test) 
#os.getcwd() #To get current working directory
titanic_test.to_csv("submission_Titanic2.csv", columns=['PassengerId','Survived'], index=False)



