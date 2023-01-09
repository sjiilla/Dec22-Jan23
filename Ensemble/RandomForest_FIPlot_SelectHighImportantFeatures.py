# -*- coding: utf-8 -*-
"""

@author: Sreenivas.J
"""

import pandas as pd
import os
from sklearn import ensemble
#from sklearn import preprocessing #Depricated  
from sklearn.impute import SimpleImputer #New version
from sklearn import model_selection
from sklearn import feature_selection
#from sklearn_pandas import CategoricalImputer

os.chdir("E:/Data Science/Data/")

titanic_train = pd.read_csv("titanic_train.csv")
titanic_train.shape
titanic_train.info()

titanic_test = pd.read_csv("titanic_test.csv")
titanic_test.shape

titanic_all = pd.concat([titanic_train, titanic_test])
titanic_all.shape
titanic_all.info()

#impute missing values for continuous features
imputable_cont_features = ['Age','Fare']
cont_imputer = SimpleImputer()
cont_imputer.fit(titanic_all[imputable_cont_features])
titanic_all[imputable_cont_features] = cont_imputer.transform(titanic_all[imputable_cont_features])

#==============================================================================
# #impute missing values for categorical features
# cat_imputer = CategoricalImputer()
# cat_imputer.fit(titanic_all['Embarked'])
# titanic_all['Embarked'] = cat_imputer.transform(titanic_all['Embarked'])
#==============================================================================

titanic_all['FamilySize'] = titanic_all['SibSp'] +  titanic_all['Parch'] + 1

def convert_family_size(size):
    if(size == 1): 
        return 'Single'
    elif(size <=3): 
        return 'Small'
    elif(size <= 6): 
        return 'Medium'
    else: 
        return 'Large'
titanic_all['FamilyCategory'] = titanic_all['FamilySize'].map(convert_family_size)

def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic_all['Title'] = titanic_all['Name'].map(extract_title)
#==============================================================================
# 
# tmp_df = titanic_all[0:titanic_train.shape[0]]
# sns.FacetGrid(tmp_df, row="Survived",size=8).map(sns.kdeplot, "FamilySize").add_legend()
# sns.factorplot(x="Title", hue="Survived", data=tmp_df, kind="count", size=6)
# sns.factorplot(x="FamilyCategory", hue="Survived", data=tmp_df, kind="count", size=6)
# sns.FacetGrid(tmp_df, row="Survived",size=8).map(sns.kdeplot, "Age").add_legend()
#==============================================================================


titanic_all.drop(['PassengerId', 'Name', 'Cabin','Ticket','Survived'], axis=1, inplace=True)

features = ['Sex', 'Embarked', 'Pclass', 'Title', 'FamilyCategory']
titanic_all = pd.get_dummies(titanic_all, columns=features)

X_train = titanic_all[0:titanic_train.shape[0]]
y_train = titanic_train['Survived']
type(X_train)

#applying feature selection algorithm to get impactful features
rf = ensemble.RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

#Feature importance plot
features = pd.DataFrame({'feature':X_train.columns, 'importance':rf.feature_importances_})
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(40, 40))

fs_model = feature_selection.SelectFromModel(rf, prefit=True)
X_train1 = fs_model.transform(X_train) 
X_train1.shape 
type(X_train1)

#build model using selected features
bagged_tree_estimator = ensemble.RandomForestClassifier(random_state=100, oob_score=True)
bagged_tree_grid = {'n_estimators':list(range(50,101,50))}
grid_bagged_tree_estimator = model_selection.GridSearchCV(bagged_tree_estimator, bagged_tree_grid, cv=10)
grid_bagged_tree_estimator.fit(X_train1, y_train)

final_model = grid_bagged_tree_estimator.best_estimator_
print(final_model.oob_score_)
print(grid_bagged_tree_estimator.best_score_)
print(grid_bagged_tree_estimator.score(X_train1, y_train))
