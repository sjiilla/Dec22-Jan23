#Functions
def add1(a,b):
  return a+b

def add2(a,b,c):
   tmp = a+b
   return tmp+c

#Defaulut values
def add3(a,b,c=20,d=40):
    tmp = a+b+c
    return tmp+d

print(add1(10,20))
print(add3(10,20))

#Add3 has 4 params
print(add3(3,4))
print(add3(3,4,10))