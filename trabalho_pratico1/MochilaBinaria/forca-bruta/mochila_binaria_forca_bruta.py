import random
import matplotlib.pyplot as plt
from itertools import product

from datetime import datetime

def knapsack_brute_force(weights, values, capacity):
    n = len(weights)
    all_values = []
    best_value = 0
    best_combination = None
    initial_time = datetime.now();
    print('inicio processamento: ', initial_time)

    for combination in product([0, 1], repeat=n):
        total_weight = sum(w for w, bit in zip(weights, combination) if bit)
        total_value = sum(v for v, bit in zip(values, combination) if bit)
        if total_weight <= capacity:
            all_values.append(total_value)
            if total_value > best_value:
                best_value = total_value
                best_combination = combination
                
             
    execution_time = datetime.now() - initial_time
    print('Tempo de execucao: ', execution_time)

    # Gráfico: distribuição dos valores válidos
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(all_values)), all_values, marker='o', linestyle='-', alpha=0.7)
    plt.axhline(best_value, color='red', linestyle='--', label=f'Melhor valor = {best_value}')
    plt.title("Valores obtidos por força bruta (Mochila Binária)")
    plt.xlabel("Combinação válida (índice)")
    plt.ylabel("Valor total da mochila")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_value, best_combination, all_values


weights = [8, 14, 11, 6, 19, 9, 12, 16, 5, 18, 7, 10, 15, 13, 17, 20, 4, 9, 11, 8, 7, 14, 10, 6, 5]
values  = [16, 30, 22, 15, 37, 19, 25, 33, 10, 36, 17, 21, 32, 26, 35, 40, 9, 18, 24, 15, 14, 29, 20, 13, 11]
capacity = 120


print("Rodando força bruta...")
best_brute_value, best_brute_comb, all_values = knapsack_brute_force(weights, values, capacity)
print(f"Melhor valor (força bruta): {best_brute_value}")
print("Combinação ótima:", best_brute_comb)