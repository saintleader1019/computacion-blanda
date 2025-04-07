import os
import tsplib95
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool
import re

# === Cargar instancia y matriz de distancias ===
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

# === Algoritmo de colonia de hormigas ===
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
                    try:
                        pij[j] = (To[current_city, next_city]**alpha) * (nij[current_city, next_city]**beta)
                    except:
                        pij[j] = 0

                total = pij.sum()
                if total == 0 or np.isnan(total):
                    next_city = np.random.choice(unvisited)  # fallback aleatorio
                else:
                    pij /= total
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

# === Leer soluciones óptimas desde archivo de texto (formato flexible) ===
def cargar_optimos_desde_txt(path_txt):
    optimos = {}
    with open(path_txt, "r") as f:
        for line in f:
            if ":" in line:
                nombre, valor = line.strip().split(":")
                nombre = nombre.strip()
                valor = valor.strip()
                valor_num = re.findall(r"[\d.]+", valor)
                if valor_num:
                    optimos[nombre] = float(valor_num[0])
    return optimos

# === Ejecutar instancia con ACO y calcular GAP ===
def run_grid_search(args):
    path_tsp, archivo_tsp, optimos_dict = args
    coords, d, problem = load_tsplib_instance(os.path.join(path_tsp, archivo_tsp))
    nombre = os.path.splitext(archivo_tsp)[0]
    optimo = optimos_dict.get(nombre, None)

    # Rango de parámetros
    n_ants_vals = [10, 20, 30]
    alpha_vals = [0.5, 1.0, 1.5]
    beta_vals  = [1.0, 2.0]
    ro_vals    = [0.1, 0.3, 0.5]
    iterations = 1000

    resultados = []
    combinaciones = list(product(n_ants_vals, alpha_vals, beta_vals, ro_vals))

    for n_ants, alpha, beta, ro in tqdm(combinaciones, desc=f"Grid search en {archivo_tsp}"):
        path, length, tiempo = run_aco(coords, d, n_ants=n_ants, alpha=alpha, beta=beta, ro=ro, iterations=iterations)
        gap = None
        if optimo:
            gap = ((length - optimo) / optimo) * 100

        resultados.append({
            "archivo": archivo_tsp,
            "dimension": problem.dimension,
            "mejor_distancia": round(length, 3),
            "optimo": optimo,
            "gap_%": round(gap, 2) if gap is not None else None,
            "tiempo_segundos": round(tiempo, 3),
            "n_ants": n_ants,
            "alpha": alpha,
            "beta": beta,
            "ro": ro,
            "iterations": iterations
        })

    df = pd.DataFrame(resultados)
    output_csv = f"gridsearch_{nombre}.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Grid search completado para {archivo_tsp}. Resultados guardados en {output_csv}")

# === Gráficos por instancia ===
def graficar_resultados_por_instancia(path_csv):
    df = pd.read_csv(path_csv)
    nombre = os.path.splitext(os.path.basename(path_csv))[0].replace("gridsearch_", "")

    # Ordenar por mejor distancia y mostrar top 5
    print(f"\n🏆 Top 5 configuraciones para {nombre} ordenadas por mejor_distancia:")
    print(df.sort_values("mejor_distancia").head(5))

    # Gráficos GAP vs cada parámetro
    parametros = ["n_ants", "alpha", "beta", "ro"]
    for param in parametros:
        plt.figure()
        df.groupby(param)["gap_%"].mean().plot(marker='o')
        plt.title(f"{nombre} - GAP promedio vs {param}")
        plt.xlabel(param)
        plt.ylabel("GAP (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"grafico_gap_vs_{param}_{nombre}.png")
        plt.close()

# === Main ===
if __name__ == "__main__":
    path_tsp = "C:/Users/santi/OneDrive/Documentos/GitHub/computacion-blanda/acoEntregaII/tsplib-master"
    ruta_solutions_txt = "C:/Users/santi/OneDrive/Documentos/GitHub/computacion-blanda/acoEntregaII/tsplib-master/solutions"
    optimos_dict = cargar_optimos_desde_txt(ruta_solutions_txt)

    # Lista de 9 instancias seleccionadas
    instancias = [
        "burma14.tsp", "ulysses16.tsp", "ulysses22.tsp",
        "berlin52.tsp", "st70.tsp", "kroA100.tsp",
        "ch150.tsp", "tsp225.tsp", "pr439.tsp"
    ]

    args_list = [(path_tsp, archivo, optimos_dict) for archivo in instancias]

    # === Para ejecutar el grid (opcional una vez completado) ===
    # with Pool(processes=min(9, os.cpu_count())) as pool:
    #     pool.map(run_grid_search, args_list)

    # === Generar gráficos por instancia ===
    for archivo in instancias:
        csv_path = f"gridsearch_{os.path.splitext(archivo)[0]}.csv"
        if os.path.exists(csv_path):
            graficar_resultados_por_instancia(csv_path)