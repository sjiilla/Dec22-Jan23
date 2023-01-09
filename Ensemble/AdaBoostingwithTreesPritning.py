import os
import pandas as pd
from sklearn import ensemble
from sklearn import tree
from sklearn import model_selection
import pydotplus
import io

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

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

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

#Note that we take entire data into consideration in boosting.
dt_estimator = tree.DecisionTreeClassifier()
#Ensemble.AdaBoostClassifier by passing base_estimator as dt_Estimator and n_estimators(no of. trees to grow) = 5
ada_tree_estimator1 = ensemble.AdaBoostClassifier(dt_estimator, n_estimators=5)
scores = model_selection.cross_val_score(ada_tree_estimator1, X_train, y_train, cv = 10)
#print(scores.mean())
ada_tree_estimator1.fit(X_train, y_train)

ada_tree_estimator1.estimators_

#extracting all the trees build by ada boost algorithm
#This tree building is only for display and understanding purpose but not requiered in reality
n_tree = 0
#Since we gave n_estimators(no of. trees to grow) = 5, it builds 5 trees
for i in ada_tree_estimator1.estimators_: 
    dot_data = io.StringIO()
    #tmp = est.tree_
    tree.export_graphviz(i, out_file = dot_data, feature_names = X_train.columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())#[0] 
    graph.write_pdf("adatree" + str(n_tree) + ".pdf")
    n_tree = n_tree + 1
    
os.getcwd()
