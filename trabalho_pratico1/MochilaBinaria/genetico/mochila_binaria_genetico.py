import random
import matplotlib.pyplot as plt
from itertools import product

from datetime import datetime

def knapsack_genetic(weights, values, capacity, pop_size=50, generations=200, mutation_rate=0.05):
    n = len(weights)
    population = [[random.randint(0, 1) for _ in range(n)] for _ in range(pop_size)]

    # Listas para armazenar estatísticas
    best_values = []
    avg_values = []
    min_values = []

    def fitness(individual):
        total_weight = sum(w for w, bit in zip(weights, individual) if bit)
        total_value = sum(v for v, bit in zip(values, individual) if bit)
        if total_weight > capacity:
            return 0  # penaliza soluções inválidas
        return total_value

    for gen in range(generations):
        # Avalia população
        scores = [fitness(ind) for ind in population]

        # Coleta estatísticas
        best_values.append(max(scores))
        avg_values.append(sum(scores) / len(scores))
        min_values.append(min(scores))

        # Elitismo: preserva o melhor indivíduo
        best_index = scores.index(max(scores))
        best_individual = population[best_index]

        # Seleção por roleta
        total_fitness = sum(scores)
        if total_fitness == 0:
            probs = [1 / pop_size] * pop_size
        else:
            probs = [s / total_fitness for s in scores]

        def select_parent():
            return population[random.choices(range(pop_size), weights=probs, k=1)[0]]

        # Nova geração
        new_population = [best_individual]  # mantém o melhor

        while len(new_population) < pop_size:
            parent1, parent2 = select_parent(), select_parent()

            # Crossover de ponto único
            point = random.randint(1, n - 1)
            child = parent1[:point] + parent2[point:]

            # Mutação
            for i in range(n):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]

            new_population.append(child)

        population = new_population

    # Melhor indivíduo final
    best_value = fitness(best_individual)
    
    
    final_time = datetime.now()
    
    print('execution time: ', final_time - initial_time)

    # --- Gráfico de convergência ---
    plt.figure(figsize=(9, 6))
    plt.plot(best_values, label='Melhor fitness', linewidth=2)
    plt.plot(avg_values, label='Fitness médio', linestyle='--')
    plt.plot(min_values, label='Pior fitness', linestyle=':')
    plt.title("Evolução do Algoritmo Genético (Mochila Binária)")
    plt.xlabel("Geração")
    plt.ylabel("Valor de fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_value, best_individual


weights = [8, 14, 11, 6, 19, 9, 12, 16, 5, 18, 7, 10, 15, 13, 17, 20, 4, 9, 11, 8, 7, 14, 10, 6, 5]
values  = [16, 30, 22, 15, 37, 19, 25, 33, 10, 36, 17, 21, 32, 26, 35, 40, 9, 18, 24, 15, 14, 29, 20, 13, 11]
capacity = 120

print("\nRodando algoritmo genético...")
initial_time = datetime.now()
print('initial time: ', initial_time)
best_ga_value, best_ga_solution = knapsack_genetic(weights, values, capacity)
print(f"\nMelhor valor (genético): {best_ga_value}")
print("Solução encontrada:", best_ga_solution)
