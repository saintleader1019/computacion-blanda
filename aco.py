import numpy as np
import matplotlib.pyplot as plt

n = 5
np.random.seed(0)
cities = np.random.uniform(0, 10, [n, 2]) 
#plt.scatter(cities[:,0], cities[:,1], c='red', alpha=0.5)
#plt.grid()
#plt.show()

# definir funcion para calcular matriz de distancias

def distances(cities):
    n = len(cities)
    d = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            #d[i,j] = np.sqrt((cities[i,0]-cities[j,0])**2 + (cities[i,1]-cities[j,1])**2) 
            d[i,j] = np.linalg.norm(cities[i]-cities[j])

    return d 

print(f'Coordinates:\n {cities}')
d = distances(cities)
print(d)

nij = 1/d
print(f'Atractiveness:\n {nij}')

To = np.ones([n,n])
print(f'Initial pheromone:\n {To}')

def sumatory(nij):
    sum = 0
    for i in range(n):
        for j in range(n):
            sum += nij[i,j]
    return sum

def aleatoryProportionalTransition(To, nij):
    n = len(To)
    p = np.zeros([n,n])
    sum = sumatory(nij)
    for i in range(n):
        for j in range(n):
            p[i,j] = nij[i,j]/sum
    return p

p = aleatoryProportionalTransition(To, nij)
print(f'Probability:\n {p}')