#In python, strings can be stored in Double Quotes/Single Quotes
nameDQ = "Sreeni DQ"
print(type(nameDQ))

nameSQ = 'Sreeni SQ'
print(type(nameSQ))

#Access string content
name = 'Sreeni DS'
print(name[0])

#slicing
print(name[0:5])

#Modify string content
name = name + 'xyz' #You can concatinate
print(name)

#Replace function
name = name.replace(name, name.upper())
print(name)

#instance: Check for instance comparision
isinstance(name, str) 
isinstance(name, int) 