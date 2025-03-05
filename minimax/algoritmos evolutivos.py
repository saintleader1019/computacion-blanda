import numpy as np
# crear funcion que convierta de binario a decimal

# def binaryToDecimal(number):
#     decimal = 0 
#     for i in range(len(number)):
#         decimal += number[i]*2**(len(number)-1-i)

#     return decimal

# b = np.array([1,1,1,1,1,1,1,1])

# xMin = 0
# xMax = 4
# d = len(b)
# vc = xMin + (xMax - xMin) / (2**d - 1) * binaryToDecimal(b)

#ejercicio 2, generar 10 soluciones (individuos) aleatorias para la var x2
# 0 <= x2 <= 1

xmin = 0
xmax = 1
d = 10
np.random.seed(0)
x1 = np.random.randint(0,2,(20,d)) #10 filas x d columnas

#calcular beneficio de el individuo y su peso
b = np.array([51,36,83,65,88,54,26,36,36,40])
p = np.array([30,38,54,21,32,33,68,30,32,38])

m = 10
c = 220

# x1 = np.array([1,1,1,0,0,0,1,1,1,0])

beneficio = np.dot(b,np.transpose(x1))
peso = np.dot(p,np.transpose(x1))

print(beneficio)
print(peso)
print(peso <= c)


