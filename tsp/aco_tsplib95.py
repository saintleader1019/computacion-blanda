import tsplib95
import numpy as np
import matplotlib.pyplot as plt
import time

def load_tsplib_instance(path):
    problem = tsplib95.load(path)
    nodes = list(problem.get_nodes())
    coords = np.array([problem.node_coords[node] for node in nodes])

    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i, j] = problem.get_weight(nodes[i], nodes[j])
            else:
                d[i, j] = np.inf
    return coords, d, problem

def run_aco(coords, d, n_ants=10, alpha=1, beta=1, ro=0.5, iterations=1):
    n = len(coords)
    nij = 1 / d
    To = np.ones([n, n])
    delta = ro

    best_path = []
    best_path_length = np.inf
    start_time = time.time()

    for _ in range(iterations):
        paths = []
        paths_length = []
        for _ in range(n_ants):
            S = np.zeros(n, dtype=bool)
            current_city = np.random.randint(n)
            S[current_city] = True
            path = [current_city]
            path_length = 0

            while not np.all(S):
                unvisited = np.where(S == False)[0]
                pij = np.zeros(len(unvisited))
                for j, next_city in enumerate(unvisited):
                    pij[j] = (To[current_city, next_city]**alpha) * (nij[current_city, next_city]**beta)
                pij /= pij.sum()
                next_city = np.random.choice(unvisited, p=pij)
                path.append(next_city)
                path_length += d[current_city, next_city]
                current_city = next_city
                S[current_city] = True

            path_length += d[current_city, path[0]]
            paths.append(path)
            paths_length.append(path_length)

            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

        # Evaporación y refuerzo de feromonas
        To *= (1 - ro)
        for path, length in zip(paths, paths_length):
            for i in range(n - 1):
                To[path[i], path[i + 1]] += delta / length
            To[path[-1], path[0]] += delta / length

    elapsed_time = time.time() - start_time
    return best_path, best_path_length, elapsed_time

def plot_solution(coords, path, title="Best Path"):
    path = path + [path[0]]  # volver al inicio
    plt.figure()
    plt.scatter(coords[:, 0], coords[:, 1], c='red', alpha=0.5)
    for i, city in enumerate(coords):
        plt.text(city[0], city[1], str(i), fontsize=9)
    plt.plot(coords[path, 0], coords[path, 1], c='blue')
    plt.title(title)
    plt.grid()
    plt.show()

coords, d, problem = load_tsplib_instance("ulysses22.tsp")
path, length, t = run_aco(coords, d, n_ants=20, alpha=1.5, beta=2.0, ro=0.3, iterations=200)
print(f"Best path length: {length:.2f}")
print(f"Execution time: {t:.2f} s")
plot_solution(coords, path)

#la mala pa daniel