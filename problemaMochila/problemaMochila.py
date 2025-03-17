import numpy as np
import matplotlib.pyplot as plt

def repair_solution(solution, benefit_values, weight_values, capacity, num_items):
    """
    Repara una solución del problema de la mochila, eliminando ítems de forma aleatoria
    hasta que el peso total no exceda la capacidad.
    """
    total_weight = np.dot(weight_values, solution)
    while total_weight > capacity:
        random_index = np.random.randint(0, num_items)
        solution[random_index] = 0
        total_weight = np.dot(weight_values, solution)
    total_benefit = np.dot(benefit_values, solution)
    return solution, total_weight, total_benefit

def uniform_crossover(parent1, parent2, num_items):
    """
    Realiza un cruce uniforme entre dos padres generando un hijo.
    Cada gen se selecciona de forma aleatoria del primer o segundo padre.
    """
    crossover_mask = np.random.randint(0, 2, num_items)
    indices_from_parent1 = crossover_mask == 1
    indices_from_parent2 = crossover_mask == 0
    child_solution = np.zeros(num_items, dtype=int)
    child_solution[indices_from_parent1] = parent1[indices_from_parent1]
    child_solution[indices_from_parent2] = parent2[indices_from_parent2]
    return child_solution

def knapsack_instances(n, w_range, b_range, alpha):
    """
    Genera una instancia del problema de la mochila.
    
    Parámetros:
    - n (int): Número de ítems.
    - w_range (tuple): Rango para los pesos, por ejemplo (1, 20) para pesos enteros uniformemente distribuidos.
    - b_range (tuple): Rango para los beneficios, por ejemplo (10, 100) para beneficios enteros uniformemente distribuidos.
    - alpha (float): Factor para definir la capacidad de la mochila relativo a la suma de los pesos.
    
    Retorna:
    - instance (dict): Diccionario con las claves 'benefit_values', 'weight_values', 'num_items' y 'capacity'.
    """
    weights = np.random.randint(w_range[0], w_range[1] + 1, n)
    benefits = np.random.randint(b_range[0], b_range[1] + 1, n)
    capacity = int(alpha * np.sum(weights))
    instance = {
        'benefit_values': benefits,
        'weight_values': weights,
        'num_items': n,
        'capacity': capacity
    }
    return instance

# ---------------------------
# Instancias "quemadas" por defecto (comentadas)
benefit_values = np.array([51, 36, 83, 65, 88, 54, 26, 36, 36, 40])
weight_values = np.array([30, 38, 54, 21, 32, 33, 68, 30, 32, 38])
num_items = 10
capacity = 220
initial_solution = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0])
# ---------------------------

# Generación de instancia mediante la función 'knapsack_instances'
# Ejemplo: 100 ítems, pesos entre 1 y 20, beneficios entre 10 y 100,
# y capacidad igual al 70% de la suma total de los pesos.
# instance = knapsack_instances(n=10, w_range=(1, 20), b_range=(10, 100), alpha=0.7)
# benefit_values = instance['benefit_values']
# weight_values  = instance['weight_values']
# num_items      = instance['num_items']
# capacity       = instance['capacity']

# Parámetros del algoritmo evolutivo
crossover_rate    = 0.9    # Porcentaje de cruzamiento (90%)
mutation_rate     = 0.1    # Tasa de mutación (10%)
population_size   = 20    # Tamaño de la población
selection_method  = "tournament"  # Puede ser "tournament" o "roulette"
penalty_method    = "repair"   # Puede ser "repair" o "penalty"
penalty_factor    = 2.0         # Factor de penalización (se descuenta penalty_factor * exceso de peso)

# Inicialización de la población
np.random.seed(0)  # Fijamos la semilla para reproducibilidad
population = np.random.randint(0, 2, [population_size, num_items])
raw_benefits = np.dot(benefit_values, population.T)
raw_weights  = np.dot(weight_values, population.T)

# Cálculo de la aptitud efectiva (fitness) según el método elegido
effective_fitness = np.zeros(population_size)

if penalty_method == "repair":
    # Se reparan los individuos infactibles
    for i in range(population_size):
        if raw_weights[i] > capacity:
            population[i, :], raw_weights[i], raw_benefits[i] = repair_solution( population[i, :], benefit_values, weight_values, capacity, num_items)
    effective_fitness = raw_benefits.copy()  # Todas son factibles
elif penalty_method == "penalty":
    # Se calcula la aptitud penalizada sin reparar
    for i in range(population_size):
        if raw_weights[i] > capacity:
            effective_fitness[i] = raw_benefits[i] - penalty_factor * (raw_weights[i] - capacity)
        else:
            effective_fitness[i] = raw_benefits[i]
else:
    raise ValueError("penalty_method debe ser 'repair' o 'penalty'.")

# Incumbente inicial (mejor individuo) según la aptitud efectiva
best_index = np.argmax(effective_fitness)

# Evolución
max_generations = 1000
plt.scatter(0, effective_fitness[best_index], c='blue')

for generation in range(max_generations):
    
    # Selección de padres según el método elegido
    if selection_method == "tournament":
        candidate_indices = np.random.choice(population_size, 4, replace=False)
        # Torneo para el primer padre
        if effective_fitness[candidate_indices[0]] > effective_fitness[candidate_indices[1]]:
            parent1_index = candidate_indices[0]
        else:
            parent1_index = candidate_indices[1]
        # Torneo para el segundo padre
        if effective_fitness[candidate_indices[2]] > effective_fitness[candidate_indices[3]]:
            parent2_index = candidate_indices[2]
        else:
            parent2_index = candidate_indices[3]
    elif selection_method == "roulette":
        total_fit = np.sum(effective_fitness)
        if total_fit == 0:
            probabilities = np.ones(population_size) / population_size
        else:
            probabilities = effective_fitness / total_fit
        parent1_index = np.random.choice(np.arange(population_size), p=probabilities)
        parent2_index = np.random.choice(np.arange(population_size), p=probabilities)
    else:
        raise ValueError("El método de selección debe ser 'tournament' o 'roulette'.")
    
    parent1 = population[parent1_index]
    parent2 = population[parent2_index]
    
    # Cruzamiento: se genera un hijo usando uniform crossover según la probabilidad definida
    if np.random.rand() <= crossover_rate:
        child = uniform_crossover(parent1, parent2, num_items)
    else:
        child = np.copy(parent1)
    
    # Mutación: con probabilidad definida
    if np.random.rand() <= mutation_rate:
        mutation_index = np.random.randint(0, num_items)
        child[mutation_index] = 0 if child[mutation_index] == 1 else 1
    
    # Evaluación del hijo según el método elegido
    child_weight = np.dot(weight_values, child)
    child_benefit = np.dot(benefit_values, child)
    if penalty_method == "repair":
        child, child_weight, child_benefit = repair_solution(child, benefit_values, weight_values, capacity, num_items)
        child_effective_fitness = child_benefit
    elif penalty_method == "penalty":
        if child_weight > capacity:
            child_effective_fitness = child_benefit - penalty_factor * (child_weight - capacity)
        else:
            child_effective_fitness = child_benefit

    # Reemplazar el peor individuo de la población si el hijo es mejor (según aptitud efectiva)
    worst_index = np.argmin(effective_fitness)
    if child_effective_fitness > effective_fitness[worst_index]:
        population[worst_index, :] = child
        raw_benefits[worst_index] = child_benefit
        raw_weights[worst_index] = child_weight
        effective_fitness[worst_index] = child_effective_fitness
    
    # Actualizar el mejor individuo (incumbente) si se encuentra uno mejor
    current_best_index = np.argmax(effective_fitness)
    if effective_fitness[current_best_index] > effective_fitness[best_index]:
        best_index = current_best_index
        plt.scatter(generation, effective_fitness[current_best_index], c='red')

plt.title("Evolución de la Aptitud Efectiva")
plt.xlabel("Generación")
plt.ylabel("Aptitud Efectiva")
plt.grid()
plt.show()

print('Mejor solución:', population[best_index],
      'Beneficio:', raw_benefits[best_index],
      'Peso:', raw_weights[best_index],
      'Aptitud efectiva:', effective_fitness[best_index])
