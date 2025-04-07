import os
import pandas as pd

BASE_DIR = "TSP_Experiment"
CSV_DIR = os.path.join(BASE_DIR, "csv")
IMG_DIR = os.path.join(BASE_DIR, "png")
README_PATH = os.path.join(BASE_DIR, "README.md")

def resumen_csv(csv_path):
    df = pd.read_csv(csv_path)
    nombre = os.path.basename(csv_path).replace("gridsearch_", "").replace(".csv", "")
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
    contenido = [
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

        "\n## 📌 Conclusiones y Recomendaciones por Instancia\n"
    ]

    for archivo in sorted(os.listdir(CSV_DIR)):
        if archivo.endswith(".csv"):
            csv_path = os.path.join(CSV_DIR, archivo)
            contenido.append(resumen_csv(csv_path))
            nombre = archivo.replace("gridsearch_", "").replace(".csv", "")
            for param in ["n_ants", "alpha", "beta", "ro"]:
                grafico_path = f"grafico_gap_vs_{param}_{nombre}.png"
                if os.path.exists(os.path.join(IMG_DIR, grafico_path)):
                    contenido.append(f"\n![GAP vs {param} - {nombre}](./png/{grafico_path})\n")

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(contenido))

if __name__ == "__main__":
    generar_readme()
    print(f"README.md generado en: {README_PATH}")
