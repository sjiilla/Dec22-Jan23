#Dictionaries are Key Value pairs
#Dictionary will be represented using {flower brackets} 

dict1 =  {'Height':10, 'Width':20}
type(dict1)

#access elements by key
print(dict1['Height'])
print(dict1['Width'])

#Alternate way to access elements by key
print(dict1.get('Height'))
print(dict1.get('Width'))

#Replace width value
dict1['Width'] = 50

#Access keys
print(dict1.keys())

#Access both key values pairs
print(dict1.items())
