import os
import pandas as pd
from sklearn import decomposition
import seaborn as sns

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

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
titanic_train1.shape
X_train.info()

#Here comes the PCA!
pca = decomposition.PCA(n_components=2)
pca.fit(X_train)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

#Transformation of PCA happens here
transformed_X_train = pca.transform(X_train)
transformed_X_train.shape

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())



y_train = titanic_train['Survived']

#Assign transformed PCA data into new data frame for visualaiztion purpose
transformed_df = pd.DataFrame(data = transformed_X_train, columns = ['pc1', 'pc2'])
#See whethere PC1 and PC2s are orthogonal are not!
sns.jointplot('pc1', 'pc2', transformed_df)

