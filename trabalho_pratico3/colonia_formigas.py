import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path
import os

# === CONFIGURAÇÕES ===
DIST_FILE = Path('distancia_matrix.csv')
dist_matrix = np.loadtxt(DIST_FILE, delimiter=',')
n_cities = dist_matrix.shape[0]

# --- Criar coordenadas artificiais (para visualização) ---
# Caso seu CSV tenha apenas distâncias (sem coordenadas)
# aqui geramos posições aleatórias para representar as cidades
np.random.seed(42)
coords = np.random.rand(n_cities, 2) * 100  # coordenadas (x, y)

# --- Funções utilitárias ---
def tour_length(tour, dist):
    """Calcula o comprimento total de uma rota."""
    return sum(dist[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))

def plot_tour(coords, tour, iteration, best_dist, save_path):
    """Gera uma figura da rota atual e salva em arquivo temporário."""
    plt.figure(figsize=(6,6))
    plt.scatter(coords[:,0], coords[:,1], c='red', s=50)
    for i in range(len(tour)):
        a, b = coords[tour[i]], coords[tour[(i+1)%len(tour)]]
        plt.plot([a[0], b[0]], [a[1], b[1]], 'b-', alpha=0.6)
    plt.title(f"Iteração {iteration} - Melhor distância: {best_dist:.2f}")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# === IMPLEMENTAÇÃO DO ACO ===
def run_aco(dist_matrix, coords, n_ants=30, n_iter=200, alpha=1, beta=5,
            evaporation=0.5, q=1.0, stable_window=50, seed=None, verbose=False):
    rng = np.random.default_rng(seed)
    n = dist_matrix.shape[0]

    # Matriz de feromônio inicial
    tau = np.full((n,n), 1.0)
    eta = np.zeros_like(dist_matrix)
    for i in range(n):
        for j in range(n):
            if i != j:
                eta[i,j] = 1.0 / dist_matrix[i,j]

    best_tour = None
    best_distance = float('inf')
    best_history = []
    best_tours_over_time = []

    for it in range(n_iter):
        all_tours = []
        all_lengths = []

        # Cada formiga constrói uma solução
        for k in range(n_ants):
            start = rng.integers(n)
            tour = [start]
            visited = {start}
            for step in range(n-1):
                i = tour[-1]
                probs = np.zeros(n)
                for j in range(n):
                    if j not in visited:
                        probs[j] = (tau[i,j]**alpha)*(eta[i,j]**beta)
                probs /= probs.sum()
                nxt = rng.choice(n, p=probs)
                tour.append(nxt)
                visited.add(nxt)
            L = tour_length(tour, dist_matrix)
            all_tours.append(tour)
            all_lengths.append(L)

            # Atualiza melhor rota global
            if L < best_distance:
                best_distance = L
                best_tour = tour.copy()

        best_history.append(best_distance)
        best_tours_over_time.append(best_tour.copy())

        # Atualização do feromônio
        tau *= (1 - evaporation)
        for k, tour in enumerate(all_tours):
            deposit = q / all_lengths[k]
            for i in range(len(tour)):
                a, b = tour[i], tour[(i+1)%len(tour)]
                tau[a,b] += deposit
                tau[b,a] += deposit

        if verbose and it % 10 == 0:
            print(f"[Iter {it}] Melhor distância até agora: {best_distance:.2f}")

    return best_tour, best_distance, best_history, best_tours_over_time


# === EXECUÇÃO DO ACO ===
best_tour, best_dist, history, tours_over_time = run_aco(
    dist_matrix, coords,
    n_ants=30,
    n_iter=200,
    evaporation=0.4,
    seed=42,
    verbose=True
)

print(f"\nMelhor distância final encontrada: {best_dist:.2f}")

# === GRÁFICO DA EVOLUÇÃO ===
plt.figure(figsize=(8,4))
plt.plot(history, 'b-')
plt.title("Evolução da Melhor Distância")
plt.xlabel("Iteração")
plt.ylabel("Distância")
plt.grid(True)
plt.tight_layout()
plt.show()

# === GERAÇÃO DO GIF ===
frames_dir = Path("aco_frames")
frames_dir.mkdir(exist_ok=True)

frames = []
for it, tour in enumerate(tours_over_time):
    path = frames_dir / f"frame_{it:03d}.png"
    plot_tour(coords, tour, it, history[it], path)
    frames.append(imageio.imread(path))

gif_path = Path("aco_evolucao.gif")
imageio.mimsave(gif_path, frames, duration=0.2)
print(f"\n✅ GIF gerado: {gif_path.resolve()}")
