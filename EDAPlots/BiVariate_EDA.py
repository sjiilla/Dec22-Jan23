import os
import pandas as pd
import seaborn as sns

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("E:/Data Science/Data")

titanic_train = pd.read_csv("titanic_train.csv")

#EDA
titanic_train.shape
titanic_train.info()

#explore bivariate relationships: categorical vs categorical 
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Sex'])
#Crosstab cab be extended to multiple columsn as well
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Pclass'])
pd.crosstab(index=titanic_train['Survived'], columns=[titanic_train['Pclass'], titanic_train['Embarked'], titanic_train['Sex']])

#margins=True gives sub total and total across cross-tab
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Sex'], margins=True)
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Pclass'], margins=True)

#hue is for further classification plotting, In this case Plot survivied for each sex.
sns.factorplot(x="Sex", hue="Survived", data=titanic_train, kind="count", size=6) 
sns.factorplot(x="Pclass", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Embarked", hue="Survived", data=titanic_train, kind="count", size=6)

#explore bivariate relationships: categorical vs continuous 
#kind="box", 
sns.factorplot(x="Fare", hue="Survived", data=titanic_train, kind="count", size=6)

#.map is a inline function like a for loop
#Survived Vs Fare
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.kdeplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.boxplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.distplot, "Fare").add_legend()

#explore bivariate relationships: continuous vs continuous 
sns.jointplot(x="SibSp", y="Parch", data=titanic_train)
