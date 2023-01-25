from sklearn import decomposition
import seaborn as sns
import pandas as pd

#Highly correlated columns (X1, X2)
df1 = pd.DataFrame({'Age':[10, 20, 30, 40],'Fare':[15, 25, 35, 45], 'SibSp':[150, 250, 350, 450],})
sns.jointplot('Age','Fare', 'SibSp', df1)
pca = decomposition.PCA(n_components=2)
pca.fit(df1)

pca.components_[0]
pca.components_[1]

##understand how much variance captured by each principal component
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())
df1_pca = pca.transform(df1)
df1_pca.shape

sns.jointplot('Age','Fare', 'SibSp')

#Not much correlated
df2 = pd.DataFrame({'Age':[10, 20, 30, 40],'Fare':[100, 5000, 1, 5]})
sns.jointplot('Age','Fare',df2)
pca = decomposition.PCA(n_components=1)
pca.fit(df2)
pca.components_[0]
#understand how much variance captured by each principal component
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())
df2_pca = pca.transform(df2)
print(df2_pca)
