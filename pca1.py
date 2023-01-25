import numpy as np

from sklearn import decomposition #PCA Package
#from sklearn.decomposition import PCA #Alternative way
import pandas as pd

#3 features with 5 records
df1= pd.DataFrame({
        'SibSp':[1,2,3,40,5],
        'FamilySize':[2,4,7,12,10],
        'Age':[350,20,50,40,50],      
        'Fare':[100,200,300,400,4]})

# =============================================================================
# df1= pd.DataFrame({
#         'Age':[10,20,30,40,50],
#         'FamilySize':[2,4,6,8,10],
#         'SibSp':[1,2,3,4,5],        
#         'Fare':[100,200,300,400,500]}) #Age, FamilySize, Fare... Are features
# 
# =============================================================================
pca = decomposition.PCA(n_components=3) #n_components means, transform the data to n dimensions.

#find eigen values and eigen vectors of covariance matrix of df1
#.fit builds PCA model for given fetures to prinicpal components
#Equation: 
#PC1 = Age*w11+FamilySize*w12+Fare*w13.....
#PC2 = Age*w21+FamilySize*w22+Fare*w23.....
#PC3 = Age*w31+FamilySize*w32+Fare*w33.....
pca.fit(df1)
#print(pca.components_)
#convert all the data points from standard basis to eigen vector basis
df1_pca = pca.transform(df1)
print(df1_pca)

#variance of data along original axes
np.var(df1.Age) + np.var(df1.FamilySize) + np.var(df1.Fare)
#variance of data along principal component axes
#show eigen values of covariance matrix in decreasing order
pca.explained_variance_

np.sum(pca.explained_variance_)

#understand how much variance captured by each principal component
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

#show the principal components
#show eigen vectors of covariance matrix of df
pca.components_[0]
pca.components_[1]
pca.components_[2]


