import os
import tsplib95
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
from itertools import product
import os
import tsplib95
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool
import re

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

# === Funciones ===
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
    df.to_csv(f"gridsearch_{nombre}.csv", index=False)
    return nombre

def graficar(csv):
    df = pd.read_csv(csv)
    nombre = csv.replace("gridsearch_", "").replace(".csv", "")
    for param in ["n_ants", "alpha", "beta", "ro"]:
        plt.figure()
        df.groupby(param)["gap_%"].mean().plot(marker='o')
        plt.title(f"GAP vs {param} - {nombre}")
        plt.xlabel(param)
        plt.ylabel("GAP (%)")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"grafico_gap_vs_{param}_{nombre}.png")
        plt.close()

def generar_reporte(csv):
    df = pd.read_csv(csv)
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
    with open(f"reporte_{nombre}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

def resumen_csv(csv):
    df = pd.read_csv(csv)
    nombre = csv.replace("gridsearch_", "").replace(".csv", "")
    best = df.sort_values("gap_%").iloc[0]
    gap_avg = df["gap_%"].mean()
    t_avg = df["tiempo_segundos"].mean()
    return f"""### 🧠 {nombre.upper()}
- 📉 GAP promedio general: {gap_avg:.2f}%
- 🥇 Mejor GAP alcanzado: {best['gap_%']:.2f}% con parámetros:
  - n_ants: {best['n_ants']}
  - alpha: {best['alpha']}
  - beta: {best['beta']}
  - ro: {best['ro']}
- ⏱️ Tiempo medio de ejecución: {t_avg:.2f} segundos

**Recomendación:** Usar esta configuración balanceada para `{nombre}`.
"""

def generar_readme():
    resumen = [
        "# Análisis de ACO sobre Instancias TSPLIB\n",
        "Este informe resume los resultados de ACO con grid search sobre 9 instancias TSPLIB.\n",
        "\n## 📖 Explicación de los Parámetros del Algoritmo ACO\n",
        "En el experimento se ajustaron distintos parámetros del algoritmo de colonia de hormigas (ACO) para resolver problemas de rutas (TSP). Cada uno de estos parámetros tiene un efecto directo sobre el comportamiento del algoritmo. A continuación se explica qué hace cada uno y cómo afectó los resultados observados:\n",

        "\n### 🐜 n_ants (Número de hormigas)",
        "**Qué significa:** Es la cantidad de soluciones simultáneas que el algoritmo construye en cada iteración.",
        "\n**Cómo influye:** Más hormigas permiten explorar más caminos posibles, pero también aumentan el tiempo de cómputo.",
        "\n**Lo que observamos:**",
        "- En problemas pequeños, usar pocas hormigas fue suficiente para encontrar buenas rutas.",
        "- En problemas más grandes, aumentar este valor ayudó a mejorar la calidad de las soluciones, aunque con mayor tiempo de ejecución.",

        "\n### 🔺 alpha (Importancia de la feromona)",
        "**Qué significa:** Controla cuánto influye el rastro de feromonas (experiencia previa) en la elección de caminos.",
        "\n**Cómo influye:** Valores altos hacen que las hormigas se guíen más por los caminos ya marcados por otras, lo que puede llevar a soluciones estables pero también a estancarse.",
        "\n**Lo que observamos:**",
        "- Un valor medio (alpha = 1.0) fue generalmente el más efectivo.",
        "- Valores bajos (alpha = 0.5) dieron buenos resultados en problemas pequeños, al favorecer la exploración.",

        "\n### 👀 beta (Importancia de la visibilidad)",
        "**Qué significa:** Mide cuánto influye la distancia entre ciudades en la decisión de las hormigas. A mayor beta, más peso se da a las ciudades más cercanas.",
        "\n**Cómo influye:** Valores altos favorecen decisiones 'cortoplacistas' (ir al nodo más cercano), mientras que valores bajos permiten rutas más globales.",
        "\n**Lo que observamos:**",
        "- En la mayoría de las instancias, un valor alto (beta = 2.0) mejoró los resultados, ya que dirigió a las hormigas por rutas más cortas de forma más efectiva.",

        "\n### 💧 ro (Tasa de evaporación)",
        "**Qué significa:** Define qué tan rápido se 'evapora' la feromona acumulada en los caminos.",
        "\n**Cómo influye:** Un valor bajo permite que las feromonas duren más tiempo, ayudando a mantener el conocimiento colectivo. Un valor alto promueve olvidar caminos rápidamente y explorar más.",
        "\n**Lo que observamos:**",
        "- Una evaporación moderada (ro = 0.3) fue útil para balancear exploración y explotación.",
        "- En instancias grandes, una evaporación baja (ro = 0.1) ayudó a mantener las rutas buenas por más tiempo y evitó perder el progreso.",

        "\n### 🧠 Conclusión general",
        "El éxito del algoritmo ACO depende de encontrar un equilibrio entre explorar nuevas rutas (exploración) y reforzar las mejores encontradas (explotación). Ajustar los parámetros correctamente puede marcar la diferencia entre una solución mediocre y una ruta casi óptima. Los resultados muestran que no hay una única combinación ganadora, pero ciertos patrones se repiten:",
        "- beta alto suele ser beneficioso.",
        "- alpha medio o bajo ayuda a explorar mejor en problemas pequeños.",
        "- ro bajo mantiene la estabilidad en problemas grandes.",
        "- n_ants debe adaptarse al tamaño del problema.\n",

        "\n## 📌 Conclusiones y Recomendaciones\n"
    ]

    for f in sorted(os.listdir()):
        if f.startswith("gridsearch_") and f.endswith(".csv"):
            resumen.append(resumen_csv(f))
            nombre = f.replace("gridsearch_", "").replace(".csv", "")
            for param in ["n_ants", "alpha", "beta", "ro"]:
                grafico_path = f"grafico_gap_vs_{param}_{nombre}.png"
                if os.path.exists(grafico_path):
                    resumen.append(f"\n![GAP vs {param} - {nombre}](./{grafico_path})\n")

    with open("README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(resumen))

# === Main ===
if __name__ == "__main__":
    path = "C:/Users/santi/OneDrive/Documentos/GitHub/computacion-blanda/acoEntregaII/tsplib-master"
    solutions = cargar_optimos(os.path.join(path, "solutions"))
    args = [(path, inst, solutions) for inst in INSTANCIAS]

    '''
    with Pool(processes=min(9, os.cpu_count())) as pool:
        nombres = pool.map(ejecutar_instancia, args)

    for nombre in nombres:
        csv = f"gridsearch_{nombre}.csv"
        graficar(csv)
        generar_reporte(csv)
    '''

    generar_readme()

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
                    next_city = np.random.choice(unvisited)
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

        To *= (1 - ro)
        for path, length in zip(paths, paths_length):
            for i in range(n - 1):
                To[path[i], path[i + 1]] += delta / length
            To[path[-1], path[0]] += delta / length

    elapsed_time = time.time() - start_time
    return best_path, best_path_length, elapsed_time

# === Leer soluciones óptimas desde archivo de texto ===
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

    print(f"\n🏆 Top 5 configuraciones para {nombre} ordenadas por mejor_distancia:")
    print(df.sort_values("mejor_distancia").head(5))

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

# === Informe Markdown por instancia ===
def analizar_influencia_parametros(path_csv):
    df = pd.read_csv(path_csv)
    nombre = os.path.splitext(os.path.basename(path_csv))[0].replace("gridsearch_", "")
    md_lines = [f"# Informe de análisis - {nombre}\n"]

    for param in ["n_ants", "alpha", "beta", "ro"]:
        agrupado = df.groupby(param)["gap_%"].mean().reset_index()
        mejor_valor = agrupado.loc[agrupado["gap_%"].idxmin()]

        md_lines.append(f"\n## Parámetro `{param}`")
        md_lines.append("\n```")
        md_lines.append(agrupado.to_string(index=False))
        md_lines.append("```")
        md_lines.append(f"\n**Mejor valor:** `{mejor_valor[param]}` con GAP promedio de **{mejor_valor['gap_%']:.2f}%**\n")

    mejor_config = df.sort_values("gap_%").head(1)
    md_lines.append("\n## ⭐ Mejor configuración global encontrada\n")
    md_lines.append("\n```")
    md_lines.append(mejor_config.to_string(index=False))
    md_lines.append("\n```")

    output_md = f"reporte_{nombre}.md"
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"✅ Informe guardado en {output_md}")
    print("\n".join(md_lines))

# === Main ===
if __name__ == "__main__":
    path_tsp = "C:/Users/santi/OneDrive/Documentos/GitHub/computacion-blanda/acoEntregaII/tsplib-master"
    ruta_solutions_txt = "C:/Users/santi/OneDrive/Documentos/GitHub/computacion-blanda/acoEntregaII/tsplib-master/solutions"
    optimos_dict = cargar_optimos_desde_txt(ruta_solutions_txt)

    instancias = [
        "burma14.tsp", "ulysses16.tsp", "ulysses22.tsp",
        "berlin52.tsp", "st70.tsp", "kroA100.tsp",
        "ch150.tsp", "tsp225.tsp", "pr439.tsp"
    ]

    args_list = [(path_tsp, archivo, optimos_dict) for archivo in instancias]

    # === Ejecutar grid search (si es necesario) ===
    # with Pool(processes=min(9, os.cpu_count())) as pool:
    #     pool.map(run_grid_search, args_list)

    # === Análisis y gráficos ===
    for archivo in instancias:
        csv_path = f"gridsearch_{os.path.splitext(archivo)[0]}.csv"
        if os.path.exists(csv_path):
            graficar_resultados_por_instancia(csv_path)
            analizar_influencia_parametros(csv_path)
