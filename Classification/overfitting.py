#Overfitting 

import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
import io
import pydotplus

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("E:/Data Science/Data")

titanic_train = pd.read_csv("titanic_train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name', 'Survived'], 1)
y_train = titanic_train['Survived']

#automate model tuning process. use grid search method
dt = tree.DecisionTreeClassifier()
param_grid = {'criterion':['entropy'],'max_depth':[3,4,5,6,7,8,9,10], 'min_samples_split':[7,8,9,10,11,12]}
dt_grid = model_selection.GridSearchCV(dt, param_grid, cv=10, n_jobs=5)
dt_grid.fit(X_train, y_train)

#This gives the scores of all param_grid comibinations. 
#dt_grid.best_score_
#Assign the best_estimator out of all
print(dt_grid.best_estimator_)
final_model = dt_grid.best_estimator_

#The goal is to build a model with high score and the diff. between best_score(validation score) and .score(train score) should be as minimum possible.
#Overfitting....
dt_grid.best_score_ #Best score for the best parameters using validation data
dt_grid.score(X_train, y_train) #.Score is on the entire train data. 

dot_data = io.StringIO() 
tree.export_graphviz(final_model, out_file = dot_data, feature_names = X_train.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("decisiont-tree-tuned1.pdf")

