import numpy as np
import matplotlib.pyplot as plt
import time

def knapsack_instances(n, w_range, b_range, alpha):
    w = np.random.randint(w_range[0], w_range[1] + 1, n)
    b = np.random.randint(b_range[0], b_range[1] + 1, n)
    C = alpha * np.sum(w)
    return w, b, C

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

def tournament_selection(fos, N):
    idx_candidates = np.random.choice(N, 4, replace=False)
    idxp1 = idx_candidates[0] if fos[idx_candidates[0]] > fos[idx_candidates[1]] else idx_candidates[1]
    idxp2 = idx_candidates[2] if fos[idx_candidates[2]] > fos[idx_candidates[3]] else idx_candidates[3]
    return idxp1, idxp2

def roulette_selection(fos, N):
    total = np.sum(fos)
    fas = fos / total
    idxp1 = np.random.choice(N, p=fas)
    idxp2 = np.random.choice(N, p=fas)
    return idxp1, idxp2

def genetic_algorithm(b, p, C, m, N, maxGen, selection_method='tournament', mutation_rate=0.1):
    np.random.seed(0)
    pop = np.random.randint(0, 2, [N, m])
    fos = np.dot(b, np.transpose(pop))
    ps = np.dot(p, np.transpose(pop))

    for i in range(N):
        if ps[i] > C:
            pop[i, :], ps[i], fos[i] = repair(pop[i, :], b, p, C, m)

    incumbent = np.argmax(fos)
    evolution = [fos[incumbent]]

    for gen in range(1, maxGen + 1):
        if selection_method == 'tournament':
            idxp1, idxp2 = tournament_selection(fos, N)
        elif selection_method == 'roulette':
            idxp1, idxp2 = roulette_selection(fos, N)

        padre1 = pop[idxp1]
        padre2 = pop[idxp2]
        hijo1 = upx(padre1, padre2, m)

        if np.random.rand() <= mutation_rate:
            pos = np.random.randint(0, m)
            hijo1[pos] = 1 - hijo1[pos]

        hijo1, ph1, foh1 = repair(hijo1, b, p, C, m)

        idx_peor = np.argmin(fos)
        if foh1 > fos[idx_peor]:
            fos[idx_peor] = foh1
            ps[idx_peor] = ph1
            pop[idx_peor, :] = hijo1

        idx_incumbent = np.argmax(fos)
        if fos[idx_incumbent] > fos[incumbent]:
            incumbent = idx_incumbent

        evolution.append(fos[incumbent])

    return evolution, fos[incumbent], ps[incumbent]

# Parámetros del experimento
instances = {
    '200 r': (200, (1, 20), (10, 100), 0.5),
    '200 p': (200, (1, 20), (10, 100), 0.5),
    '500 r': (500, (1, 20), (10, 100), 0.5),
    '500 p': (500, (1, 20), (10, 100), 0.5),
    '750 *': (750, (1, 20), (10, 100), 0.5),
    '1000 *': (1000, (1, 20), (10, 100), 0.5)
}

results = []

plt.figure(figsize=(10, 6))

for instance_name, params in instances.items():
    n, w_range, b_range, alpha = params
    w, b, C = knapsack_instances(n, w_range, b_range, alpha)
    m = n
    N = 50  # Tamaño de la población
    maxGen = 200
    selection_method = 'tournament' if 'r' in instance_name else 'roulette'
    mutation_rate = 0.05

    start_time = time.time()
    evolution, best_fo, best_p = genetic_algorithm(b, w, C, m, N, maxGen, selection_method, mutation_rate)
    elapsed_time = time.time() - start_time

    results.append((instance_name, best_fo, elapsed_time))
    plt.plot(evolution, label=f'Instancia {instance_name}')

plt.xlabel("Generaciones")
plt.ylabel("Mejor Beneficio")
plt.title("Evolución del Beneficio en el Algoritmo Genético")
plt.legend()
plt.grid()
plt.show()

# Mostrar resultados en una tabla
print("\nResultados del Experimento")
print("+--------------+-------+------------+")
print("| Instance     | Sol*  | Time (s)   |")
print("+--------------+-------+------------+")
for res in results:
    print(f"| {res[0]:<12} | {res[1]:<5} | {res[2]:<10.4f} |")
print("+--------------+-------+------------+")