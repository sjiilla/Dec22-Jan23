#In this example, we will see how to have a conditions on sub set of data
import pandas as pd

col1 = [10,20,30,40]
col2 = ['abc','def','xyz','pqr']
col3 = [0,0,0,0]

#creating data frame, This is an alternate way instead of reading csv directly
df1 = pd.DataFrame({'pid':col1, 'pname':col2, 'survived':col3})
df1.shape
df1.info()
df1.describe()
df1.head(2)
df1.tail(2)

df1['col4'] = 0

#access frame content by column/columns
df1.pid
df1['pid']
df1[['pid','pname']]
df1[[0,1]]

#dropping a column
#drop has inplace parameter, which means whther to create a new frame or same old frame
df2 = df1.drop('survived',1)
print(df2)


#slicing rows of frame
df1[0:2]
df1[0:4]
df1[0:]
df1[:2]

#filtering rows of dataframe by condition
type(df1.pid > 20)
print(type(df1.pid > 20))
df1[df1.pid>20]
print(df1)

#selecting subsets of rows and columns
df1.iloc[0:2,]
df1.iloc[[0,2],]
df1.iloc[0:2,0]
df1.iloc[0:2,[0,2]]
df1.loc[0:2,['pname']]

#grouping data in data frames
df1.groupby('pid').size()