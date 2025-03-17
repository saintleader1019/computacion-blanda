import numpy as np
import matplotlib.pyplot as plt

def knapsack_instances(n, w_range, b_range, alpha):
    """
    Generates an instance of the knapsack problem.
    Parameters:
    - n (int): Number of items.
    - w_range: U [1,20]
    - b_range : U [10,100]
    - alpha : factor to define the knapsack capacity relative to the sum of item weights.
    """
    w = np.random.randint(w_range[0], w_range[1] + 1, n)
    b = np.random.randint(b_range[0], b_range[1] + 1, n)
    C = alpha * np.sum(w)
    return w, b, int(C)

def repair(x, b, p, C, m):
    pi = np.dot(p, x)
    while pi > C:
        pos = np.random.randint(0, m)
        x[pos] = 0
        pi = np.dot(p, x)
    foi = np.dot(b, x)
    return x, pi, foi

def upx(p1, p2, m):
    mask = np.random.randint(0, 2, m)
    m1 = mask == 1
    m0 = mask == 0
    h1 = np.zeros(m, dtype=int)
    h1[m1] = p1[m1]
    h1[m0] = p2[m0]
    return h1

def roulette_selection(fos, N):
    total = np.sum(fos)
    probabilities = fos / total
    selected_indices = np.random.choice(N, size=N, p=probabilities)
    return selected_indices

def roulette_selection_normalized(fos, N):
    if np.all(fos == fos[0]):  # Si todos los valores son iguales
        probabilities = np.ones(N) / N  # Probabilidades uniformes
    else:
        fos_normalized = fos - np.min(fos)  # Normalización
        total = np.sum(fos_normalized)
        probabilities = fos_normalized / total
    selected_indices = np.random.choice(N, size=N, p=probabilities)
    return selected_indices

# Generar instancia del problema
m = 10
w, b, C = knapsack_instances(m, (1, 20), (10, 100), 0.5)

N = 20
crossover_rate = 0.8
mutation_rate = 0.1

np.random.seed(0)
pop = np.random.randint(0, 2, [N, m])

fos = np.dot(b, np.transpose(pop))
ps = np.dot(w, np.transpose(pop))

for i in range(N):
    if ps[i] > C:
        pop[i, :], ps[i], fos[i] = repair(pop[i, :], b, w, C, m)

incertidumbre = np.argmax(fos)

maxGen = 1000
plt.scatter(0, fos[incertidumbre], c='blue')

for gen in range(maxGen):
    selected_indices = roulette_selection(fos, N)
    
    new_pop = []
    new_fos = []
    new_ps = []
    
    for i in range(0, N, 2):
        if i+1 >= N:
            break
        
        idxp1 = selected_indices[i]
        idxp2 = selected_indices[i+1]
        
        if np.random.rand() < crossover_rate:
            hijo1 = upx(pop[idxp1], pop[idxp2], m)
            hijo2 = upx(pop[idxp2], pop[idxp1], m)
        else:
            hijo1 = pop[idxp1]
            hijo2 = pop[idxp2]
        
        if np.random.rand() < mutation_rate:
            pos = np.random.randint(0, m)
            hijo1[pos] = 1 - hijo1[pos]
        
        if np.random.rand() < mutation_rate:
            pos = np.random.randint(0, m)
            hijo2[pos] = 1 - hijo2[pos]
        
        hijo1, ph1, foh1 = repair(hijo1, b, w, C, m)
        hijo2, ph2, foh2 = repair(hijo2, b, w, C, m)
        
        new_pop.extend([hijo1, hijo2])
        new_fos.extend([foh1, foh2])
        new_ps.extend([ph1, ph2])
    
    pop = np.array(new_pop)
    fos = np.array(new_fos)
    ps = np.array(new_ps)
    
    idxincum = np.argmax(fos)
    if fos[idxincum] > fos[incertidumbre]:
        incertidumbre = idxincum
        plt.scatter(gen, fos[idxincum], c='red')

plt.grid()
plt.show()
print('Mejor solución:', pop[incertidumbre], 'FO:', fos[incertidumbre], 'P:', ps[incertidumbre])
