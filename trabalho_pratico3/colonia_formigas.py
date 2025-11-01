"""
Implementação do Algoritmo de Colônia de Formigas (ACO)
para o problema do Caixeiro Viajante (TSP).

O script:
- Lê o arquivo distancia_matrix.csv
- Executa o ACO para encontrar a menor rota
- Gera gráficos e um GIF da evolução
- Analisa o impacto do número de formigas e da taxa de evaporação
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path
from statistics import mean, stdev
import os
import time

# =====================================================
# === CONFIGURAÇÕES INICIAIS ==========================
# =====================================================
DIST_FILE = Path("distancia_matrix.csv")  # arquivo fornecido
OUTPUT_DIR = Path("resultados_aco")
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)  # reprodutibilidade

# =====================================================
# === FUNÇÕES AUXILIARES ==============================
# =====================================================
def tour_length(tour, dist):
    """Calcula o comprimento total de uma rota."""
    return sum(dist[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def convergence_iteration(best_history, stable_window=100):
    """Verifica a partir de qual iteração a solução se manteve estável."""
    if len(best_history) < stable_window:
        return None
    for i in range(len(best_history) - stable_window + 1):
        window = best_history[i:i + stable_window]
        if all(abs(x - window[0]) < 1e-9 for x in window):
            return i
    return None


def plot_tour(coords, tour, iteration, best_dist, save_path):
    """Plota a rota atual e salva a imagem."""
    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=60)
    for i in range(len(tour)):
        a, b = coords[tour[i]], coords[tour[(i + 1) % len(tour)]]
        plt.plot([a[0], b[0]], [a[1], b[1]], 'b-', alpha=0.7)
    plt.title(f"Iteração {iteration} - Melhor distância: {best_dist:.2f}")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =====================================================
# === IMPLEMENTAÇÃO DO ACO ============================
# =====================================================
def run_aco(dist_matrix, coords, n_ants=30, n_iter=1000,
            alpha=1.0, beta=5.0, evaporation=0.5,
            q=1.0, stable_window=100, seed=None,
            verbose=False):
    """
    Executa o ACO e retorna:
    - melhor_tour
    - melhor_distância
    - histórico da melhor distância
    - lista das melhores rotas a cada iteração
    """
    rng = np.random.default_rng(seed)
    n = dist_matrix.shape[0]

    eta = np.zeros_like(dist_matrix)
    for i in range(n):
        for j in range(n):
            if i != j:
                eta[i, j] = 1.0 / dist_matrix[i, j]

    tau = np.full((n, n), 1.0)
    best_tour, best_distance = None, float('inf')
    best_history, best_tours_over_time = [], []

    for it in range(n_iter):
        all_tours, all_lengths = [], []
        for k in range(n_ants):
            start = rng.integers(n)
            tour = [start]
            visited = {start}

            for step in range(n - 1):
                i = tour[-1]
                probs = np.zeros(n)
                for j in range(n):
                    if j not in visited:
                        probs[j] = (tau[i, j] ** alpha) * (eta[i, j] ** beta)
                probs /= probs.sum()
                nxt = rng.choice(n, p=probs)
                tour.append(nxt)
                visited.add(nxt)

            L = tour_length(tour, dist_matrix)
            all_tours.append(tour)
            all_lengths.append(L)

            if L < best_distance:
                best_distance = L
                best_tour = tour.copy()

        best_history.append(best_distance)
        best_tours_over_time.append(best_tour.copy())

        tau *= (1 - evaporation)
        for k, tour in enumerate(all_tours):
            deposit = q / all_lengths[k]
            for i in range(len(tour)):
                a, b = tour[i], tour[(i + 1) % len(tour)]
                tau[a, b] += deposit
                tau[b, a] += deposit

        if verbose and it % 10 == 0:
            print(f"[Iter {it}] Melhor distância até agora: {best_distance:.2f}")

        conv_it = convergence_iteration(best_history, stable_window)
        if conv_it is not None:
            if verbose:
                print(f"Convergência detectada na iteração {it}.")
            break

    return best_tour, best_distance, best_history, best_tours_over_time


# =====================================================
# === LEITURA DO ARQUIVO E GERAÇÃO DE COORDENADAS ====
# =====================================================
print("Lendo matriz de distâncias...")
dist_matrix = np.loadtxt(DIST_FILE, delimiter=',')
n_cities = dist_matrix.shape[0]
coords = np.random.rand(n_cities, 2) * 100  # posições aleatórias para visualização

# =====================================================
# === EXECUÇÃO PRINCIPAL DO ACO =======================
# =====================================================
print("\nExecutando ACO principal...")
best_tour, best_dist, history, tours_over_time = run_aco(
    dist_matrix, coords, n_ants=30, n_iter=500,
    evaporation=0.4, stable_window=100,
    seed=42, verbose=True
)

print(f"\nMelhor distância final: {best_dist:.2f}")

# === GRÁFICO DE EVOLUÇÃO ===
plt.figure(figsize=(8, 4))
plt.plot(history, 'b-')
plt.title("Evolução da Melhor Distância")
plt.xlabel("Iteração")
plt.ylabel("Distância")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "evolucao_distancia.png")
plt.show()

# === GIF DA EVOLUÇÃO ===
print("\nGerando GIF da evolução das rotas...")
frames_dir = OUTPUT_DIR / "frames"
frames_dir.mkdir(exist_ok=True)
frames = []

for it, tour in enumerate(tours_over_time):
    path = frames_dir / f"frame_{it:03d}.png"
    plot_tour(coords, tour, it, history[it], path)
    frames.append(imageio.imread(path))

gif_path = OUTPUT_DIR / "aco_evolucao.gif"
imageio.mimsave(gif_path, frames, duration=0.2)
print(f"✅ GIF salvo em: {gif_path.resolve()}")

# =====================================================
# === EXPERIMENTO 1: NÚMERO DE FORMIGAS ===============
# =====================================================
print("\n[Experimento 1] Impacto do número de formigas...")
ant_counts = [5, 10, 20, 30, 40]
trials = 10
results_ant = {}

for ants in ant_counts:
    conv_iters = []
    print(f"  -> Testando {ants} formigas...")
    for t in range(trials):
        _, _, hist, _ = run_aco(dist_matrix, coords, n_ants=ants, n_iter=1000, evaporation=0.4)
        conv = convergence_iteration(hist)
        conv_iters.append(conv)
    conv_values = [c for c in conv_iters if c is not None]
    results_ant[ants] = {
        'mean': mean(conv_values) if conv_values else None,
        'stdev': stdev(conv_values) if len(conv_values) > 1 else None
    }

# Gráfico
ants_plot, means, stds = [], [], []
for ants in ant_counts:
    r = results_ant[ants]
    if r['mean'] is not None:
        ants_plot.append(ants)
        means.append(r['mean'])
        stds.append(r['stdev'] or 0)

plt.figure(figsize=(8, 4))
plt.errorbar(ants_plot, means, yerr=stds, marker='o', capsize=4)
plt.title("Impacto do Número de Formigas na Convergência")
plt.xlabel("Número de Formigas")
plt.ylabel("Iterações até Convergência")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "impacto_formigas.png")
plt.show()

# =====================================================
# === EXPERIMENTO 2: TAXA DE EVAPORAÇÃO ===============
# =====================================================
print("\n[Experimento 2] Impacto da taxa de evaporação...")
evap_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
trials = 5
results_evap = {}

for evap in evap_rates:
    conv_iters = []
    print(f"  -> Testando evaporação = {evap}...")
    for t in range(trials):
        _, _, hist, _ = run_aco(dist_matrix, coords, n_ants=30, n_iter=1000, evaporation=evap)
        conv = convergence_iteration(hist)
        conv_iters.append(conv)
    conv_values = [c for c in conv_iters if c is not None]
    results_evap[evap] = {
        'mean': mean(conv_values) if conv_values else None,
        'stdev': stdev(conv_values) if len(conv_values) > 1 else None
    }

# Gráfico
evap_plot, means_e, stds_e = [], [], []
for evap in evap_rates:
    r = results_evap[evap]
    if r['mean'] is not None:
        evap_plot.append(evap)
        means_e.append(r['mean'])
        stds_e.append(r['stdev'] or 0)

plt.figure(figsize=(8, 4))
plt.errorbar(evap_plot, means_e, yerr=stds_e, marker='o', capsize=4)
plt.title("Impacto da Taxa de Evaporação na Convergência")
plt.xlabel("Taxa de Evaporação")
plt.ylabel("Iterações até Convergência")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "impacto_evaporacao.png")
plt.show()

# =====================================================
# === RESUMO FINAL ===================================
# =====================================================
print("\n✅ Todos os experimentos concluídos.")
print(f"Resultados salvos em: {OUTPUT_DIR.resolve()}")
