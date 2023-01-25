from sklearn import decomposition, preprocessing
import pandas as pd

#PCA without standardization
sqft = [1000, 600, 1200, 1400] #sft units
n_rooms = [3, 2, 2, 10] #Integer number
price = [1000000, 700000, 1200000, 1500000] #Currency $$
house_data = pd.DataFrame({'sqft':sqft, 'n_rooms':n_rooms, 'price':price})
print(house_data)
pca = decomposition.PCA()
pca.fit(house_data)
pca.explained_variance_
pca.explained_variance_ratio_
#array([9.99999971e-01, 2.94477497e-08, 1.24414873e-11])
#array([0.88541483, 0.11358937, 0.0009958 ])

pca.explained_variance_ratio_.cumsum()

#PCA with standardization
scaler = preprocessing.StandardScaler()
scaler.fit(house_data)
scaled_house_data = scaler.transform(house_data)
print(scaled_house_data)
pca_s = decomposition.PCA()
pca_s.fit(scaled_house_data)
pca_s.explained_variance_
pca_s.explained_variance_ratio_
pca_s.explained_variance_ratio_.cumsum()
