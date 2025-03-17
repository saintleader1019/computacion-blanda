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

# Datos del problema
benefit_values = np.array([51, 36, 83, 65, 88, 54, 26, 36, 36, 40])
weight_values = np.array([30, 38, 54, 21, 32, 33, 68, 30, 32, 38])
num_items = 10
capacity = 220
initial_solution = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0])

initial_benefit = np.dot(benefit_values, initial_solution)
initial_weight = np.dot(weight_values, initial_solution)

# Inicialización de la población
np.random.seed(0)  # Fijamos la semilla para reproducibilidad
population_size = 20
population = np.random.randint(0, 2, [population_size, num_items])

population_benefits = np.dot(benefit_values, population.T)
population_weights = np.dot(weight_values, population.T)

# Reparación de individuos infactibles
for i in range(population_size):
    if population_weights[i] > capacity:
        population[i, :], population_weights[i], population_benefits[i] = repair_solution(
            population[i, :], benefit_values, weight_values, capacity, num_items)

# Incumbente inicial (mejor individuo)
best_index = np.argmax(population_benefits)

# Selección de padres (por torneo) y evolución
max_generations = 1000
plt.scatter(0, population_benefits[best_index], c='blue')

for generation in range(max_generations):
    candidate_indices = np.random.choice(population_size, 4, replace=False)
    
    # Torneo para el primer padre
    if population_benefits[candidate_indices[0]] > population_benefits[candidate_indices[1]]:
        parent1_index = candidate_indices[0]
    else:
        parent1_index = candidate_indices[1]
    
    # Torneo para el segundo padre
    if population_benefits[candidate_indices[2]] > population_benefits[candidate_indices[3]]:
        parent2_index = candidate_indices[2]
    else:
        parent2_index = candidate_indices[3]
    
    parent1 = population[parent1_index]
    parent2 = population[parent2_index]
    
    # Cruce: se genera un hijo usando uniform crossover
    child = uniform_crossover(parent1, parent2, num_items)
    
    # Mutación: con probabilidad de 10%
    mutation_probability = 0.1
    if np.random.rand() <= mutation_probability:
        mutation_index = np.random.randint(0, num_items)
        child[mutation_index] = 0 if child[mutation_index] == 1 else 1
    
    # Reparar la nueva solución
    child, child_weight, child_benefit = repair_solution(child, benefit_values, weight_values, capacity, num_items)
    # Re-cálculo (por si acaso)
    child_benefit = np.dot(benefit_values, child)
    child_weight = np.dot(weight_values, child)
    
    # Reemplazar el peor individuo de la población si el hijo es mejor
    worst_index = np.argmin(population_benefits)
    if child_benefit > population_benefits[worst_index]:
        population_benefits[worst_index] = child_benefit
        population_weights[worst_index] = child_weight
        population[worst_index, :] = child
    
    # Actualizar el mejor individuo (incumbente) si se encuentra uno mejor
    current_best_index = np.argmax(population_benefits)
    if population_benefits[current_best_index] > population_benefits[best_index]:
        best_index = current_best_index
        plt.scatter(generation, population_benefits[current_best_index], c='red')

plt.grid()
plt.show()
print('Mejor solución:', population[best_index], 'Beneficio:', population_benefits[best_index], 'Peso:', population_weights[best_index])
