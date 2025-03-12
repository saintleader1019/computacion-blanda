import numpy as np
import matplotlib.pyplot as plt

n = 50
np.random.seed(0)
cities = np.random.uniform(0, 10, [n, 2]) 

# definir funcion para calcular matriz de distancias
def distances(cities):
    n = len(cities)
    d = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            d[i, j] = np.linalg.norm(cities[i] - cities[j])
    # Evitar división por cero asignando un valor muy alto (o infinito) en la diagonal
    np.fill_diagonal(d, np.inf)
    return d

d = distances(cities)
print(d)

nij = 1/d
print(f'Atractiveness:\n {nij}')

To = np.ones([n,n])
print(f'Initial pheromone:\n {To}')

ro = 0.5 # Evaporation rate
delta = ro #factor de refuerzo

maxIter = 1
n_ants = 1
alpha = 1
beta = 1

for iter in range(maxIter):
    path=[]
    path_length = []
    for ant in range(n_ants):
        #setting S set unvisited cities
        S = np.zeros(n)  #n number of cities
        current_city = np.random.randint(n)
        print(f'Current city: {current_city}')
        S[current_city] = True
        print(f'Set of unvisited cities: {S}')
        path=[current_city]
        path_length = 0
        while False in S:
            unvisited = np.where(S == False)[0] #firt column of the matrix
            pij = np.zeros(len(unvisited))
            for j, univisted_city in enumerate(unvisited):
                pij[j] = (To[current_city, univisted_city]**alpha)*(nij[current_city, univisted_city]**beta)
            pij/=np.sum(pij)
            print(f'Probabilities: {pij}')

            next_city = np.random.choice(unvisited, p=pij)
            print(f'Next city: {next_city}')
            path.append(next_city)
            path_length += d[current_city, next_city]
            print(f'Path: {path}')
            print(f'Path length: {path_length/10}')
            #update
            current_city = next_city
            S[current_city] = True
        path.append(path[0])
        path_length += d[current_city, path[0]]
        print(f'Path: {path}')
        print(f'Path length: {path_length/10}')
        
#graficar puntos en el plot
plt.scatter(cities[:,0], cities[:,1], c='red', alpha=0.5)
for i, city in enumerate(cities):
    plt.text(city[0], city[1], str(i))
plt.grid()
plt.plot(cities[path,0], cities[path,1], c='blue')
plt.show()


# def sumatory(nij):
#     sum = 0
#     for i in range(n):
#         for j in range(n):
#             sum += nij[i,j]
#     return sum

# def aleatoryProportionalTransition(To, nij):
#     n = len(To)
#     p = np.zeros([n,n])
#     sum = sumatory(nij)
#     for i in range(n):
#         for j in range(n):
#             p[i,j] = nij[i,j]/sum
#     return p

# p = aleatoryProportionalTransition(To, nij)
# print(f'Probability:\n {p}')