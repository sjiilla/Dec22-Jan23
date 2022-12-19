# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:31:01 2018

@author: Sreenivas.J
"""

#Basic Datatypes
x = 10
print (type(x))

y = "sreeni"
y =  "sreeni" +  "DS"

x = 10.2
type(x)
print (type(x))

#Normal Data type
x = 10 + 15
print (type(x))

#j.... Complex data type
x = 'Sreeni'
print (type(x))

x = True #Should be Camel Notation
print (type(x))

#Complex Data types which comes with the inbuilt Python language
#CONTAINERS
#sets allows you to do operations such as intersection , union , difference etc...
set1 = {100,200,300,400,500} #Set {}
print (type(set1))

#Dictionary
dictionary1 = {'a':100,'b':200}
print (type(dictionary1))



list1 = [100,200,300,400,500] #List []
print (type(list1))
list1.append(600)

tuple1 = (100,200,300,400,500) #Tuple () #Read only
print (type(tuple1))

#For Array and DataFrames, we need to explicitly add packages/name spaces
#NumPy - Numerical Python 
#For Arrays

#Pandas
#For Data Frames

#This range is not working in 
list2 = range(1,10,1)
print (type(list2))

#Sliced access to elements of a list
list1 = [100,200,300,400,500] #List []
print(list1[0])
list1[-1]
list1[0:2] #Slicing is like sub string
list1[0:] #Give me from 0 index to ALL
list1[:3] #Give me from 0 index to 2 index
list1[0::2]# It's like start from 0 index to end with skipping 2 indexs. It's like get me alternate indexes
list1[0] = 100
     
#MOdifying the contents of list1

#Sort the elements of list1
list1.sort() #Sort works only on when all elemets are homogenious
print(list1)

#iterate thorughg... For loop
for x in list1:
    print(x)

     
