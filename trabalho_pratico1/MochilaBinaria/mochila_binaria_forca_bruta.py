import random
import matplotlib.pyplot as plt
from itertools import product

def knapsack_brute_force(weights, values, capacity):
    n = len(weights)
    all_values = []
    best_value = 0
    best_combination = None

    for combination in product([0, 1], repeat=n):
        total_weight = sum(w for w, bit in zip(weights, combination) if bit)
        total_value = sum(v for v, bit in zip(values, combination) if bit)
        if total_weight <= capacity:
            all_values.append(total_value)
            if total_value > best_value:
                best_value = total_value
                best_combination = combination

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


weights = [2, 3, 4, 5, 9, 7, 6]
values = [3, 4, 5, 8, 10, 6, 7]
capacity = 15

print("Rodando força bruta...")
best_brute_value, best_brute_comb, all_values = knapsack_brute_force(weights, values, capacity)
print(f"Melhor valor (força bruta): {best_brute_value}")
print("Combinação ótima:", best_brute_comb)