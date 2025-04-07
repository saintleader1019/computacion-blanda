# === Configuración de carpetas base ===
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

BASE_DIR = "TSP_Experiment"
CSV_DIR = os.path.join(BASE_DIR, "csv")
IMG_DIR = os.path.join(BASE_DIR, "png")
MD_DIR = os.path.join(BASE_DIR, "md")
README_PATH = os.path.join(BASE_DIR, "README.md")

for path in [BASE_DIR, CSV_DIR, IMG_DIR, MD_DIR]:
    os.makedirs(path, exist_ok=True)

# === Configuración ===
INSTANCIAS = [
    "burma14.tsp", "ulysses16.tsp", "ulysses22.tsp",
    "berlin52.tsp", "st70.tsp", "kroA100.tsp",
    "ch150.tsp", "tsp225.tsp", "pr439.tsp"
]
PARAMS = {
    "n_ants": [10, 20, 30],
    "alpha": [0.5, 1.0, 1.5],
    "beta": [1.0, 2.0],
    "ro": [0.1, 0.3, 0.5],
    "iterations": 1000
}

def cargar_tsplib(path):
    problem = tsplib95.load(path)
    coords = np.array([problem.node_coords[node] for node in problem.get_nodes()])
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = problem.get_weight(i + 1, j + 1) if i != j else np.inf
    return coords, d, problem

def run_aco(coords, d, n_ants, alpha, beta, ro, iterations):
    n = len(coords)
    nij = 1 / d
    To = np.ones([n, n])
    delta = ro
    best_path, best_length = [], np.inf
    start = time.time()

    for _ in range(iterations):
        paths, lengths = [], []
        for _ in range(n_ants):
            S = np.zeros(n, dtype=bool)
            current = np.random.randint(n)
            S[current] = True
            path = [current]
            length = 0

            while not np.all(S):
                unvisited = np.where(S == False)[0]
                pij = np.array([
                    (To[current, j]**alpha)*(nij[current, j]**beta) for j in unvisited
                ])
                pij = pij / pij.sum() if pij.sum() > 0 else np.ones_like(pij)/len(pij)
                next_city = np.random.choice(unvisited, p=pij)
                path.append(next_city)
                length += d[current, next_city]
                S[next_city] = True
                current = next_city

            length += d[current, path[0]]
            paths.append(path)
            lengths.append(length)
            if length < best_length:
                best_path, best_length = path, length

        To *= (1 - ro)
        for path, l in zip(paths, lengths):
            for i in range(n - 1):
                To[path[i], path[i+1]] += delta / l
            To[path[-1], path[0]] += delta / l

    return best_path, best_length, time.time() - start

def cargar_optimos(path):
    optimos = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                k, v = line.strip().split(':')
                nums = re.findall(r"[\d.]+", v)
                if nums:
                    optimos[k.strip()] = float(nums[0])
    return optimos

def ejecutar_instancia(args):
    path_dir, archivo, optimos = args
    coords, d, problem = cargar_tsplib(os.path.join(path_dir, archivo))
    nombre = archivo.replace(".tsp", "")
    optimo = optimos.get(nombre)

    resultados = []
    for n_ants, alpha, beta, ro in product(PARAMS["n_ants"], PARAMS["alpha"], PARAMS["beta"], PARAMS["ro"]):
        _, dist, t = run_aco(coords, d, n_ants, alpha, beta, ro, PARAMS["iterations"])
        gap = ((dist - optimo)/optimo)*100 if optimo else None
        resultados.append({
            "archivo": archivo,
            "dimension": problem.dimension,
            "mejor_distancia": round(dist, 3),
            "optimo": optimo,
            "gap_%": round(gap, 2) if gap is not None else None,
            "tiempo_segundos": round(t, 3),
            "n_ants": n_ants, "alpha": alpha, "beta": beta, "ro": ro
        })

    df = pd.DataFrame(resultados)
    df.to_csv(os.path.join(CSV_DIR, f"gridsearch_{nombre}.csv"), index=False)
    return nombre

def graficar(csv):
    df = pd.read_csv(os.path.join(CSV_DIR, csv))
    nombre = csv.replace("gridsearch_", "").replace(".csv", "")
    for param in ["n_ants", "alpha", "beta", "ro"]:
        plt.figure()
        df.groupby(param)["gap_%"].mean().plot(marker='o')
        plt.title(f"GAP vs {param} - {nombre}")
        plt.xlabel(param)
        plt.ylabel("GAP (%)")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, f"grafico_gap_vs_{param}_{nombre}.png"))
        plt.close()

def generar_reporte(csv):
    df = pd.read_csv(os.path.join(CSV_DIR, csv))
    nombre = csv.replace("gridsearch_", "").replace(".csv", "")
    md = [f"# Informe de análisis - {nombre}\n"]
    for param in ["n_ants", "alpha", "beta", "ro"]:
        tabla = df.groupby(param)["gap_%"].mean().reset_index()
        mejor = tabla.loc[tabla["gap_%"].idxmin()]
        md.append(f"\n## {param}\n\n```\n{tabla.to_string(index=False)}\n```\n\n**Mejor:** `{mejor[param]}` con {mejor['gap_%']:.2f}%\n")
    best = df.sort_values("gap_%").head(1)
    md.append("\n## ⭐ Mejor configuración global\n\n```")
    md.append(best.to_string(index=False))
    md.append("```\n")
    with open(os.path.join(MD_DIR, f"reporte_{nombre}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

def main():
    path = "C:/Users/santi/OneDrive/Documentos/GitHub/computacion-blanda/acoEntregaII/tsplib-master"
    solutions = cargar_optimos(os.path.join(path, "solutions"))
    args = [(path, inst, solutions) for inst in INSTANCIAS]

    with Pool(processes=min(9, os.cpu_count())) as pool:
        nombres = pool.map(ejecutar_instancia, args)

    for nombre in nombres:
        csv = f"gridsearch_{nombre}.csv"
        graficar(csv)
        generar_reporte(csv)

if __name__ == "__main__":
    main()
