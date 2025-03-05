import numpy as np

# x = [ [0],[1],[0],[1]]

# x = np.array(x)
# c = np.array([10,12,6,8])

# fo = np.dot(np.transpose(c),x)
# print(fo)

# ele = c * x
# print(ele)


# fo1 = np.dot(c,x)
# print(fo1)
# print(fo1.shape)

#-------------------------------------------------------------------------------
#minimizar
# x = np.array([[0],[1],[0],[1]])
# c = np.array([[10],[12],[6],[8]])

# fo = np.dot(np.transpose(c),x)
# print(fo)
# print(fo.shape)

#--------------------------------------------------------------------------------
#maximizar
x = np.array([0,1,0,1])
x1 = np.array([1,0,0,0])
x2 = np.array([1,1,0,0])

c = np.array([10,12,6,8])
A = np.array([10,10,7,6])

fo = np.dot(np.transpose(c),x1)

costo = np.dot(A,x)
costo1 = np.dot(A,x1)

sols = np.array([x,x1,x2])
profit = np.dot(c, np.transpose(sols))
print(profit)
print(profit.shape)

costo3 = np.dot(A,np.transpose(sols))
print(costo3)
print(costo3.shape)