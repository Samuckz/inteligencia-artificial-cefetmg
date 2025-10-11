import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# --- Parâmetros do Algoritmo Genético ---
POP_SIZE = 50
NUM_GENERATIONS = 25
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
BOUNDS = [0, 10]  # Limites para x1 e x2
DIM = 2  # n = 2

# --- Função Alpine 2 ---
def falpine2(x):
    return np.prod(np.sqrt(x) * np.sin(x))

# --- Inicialização da População ---
def initialize_population(size, dim):
    return np.random.uniform(BOUNDS[0], BOUNDS[1], (size, dim))

# --- Avaliação da População ---
def evaluate_population(pop):
    return np.array([falpine2(ind) for ind in pop])

# --- Seleção por Torneio ---
def tournament_selection(pop, fitness, k=3):
    selected = []
    for _ in range(len(pop)):
        aspirants_idx = np.random.randint(0, len(pop), k)
        best_idx = aspirants_idx[np.argmax(fitness[aspirants_idx])]
        selected.append(pop[best_idx])
    return np.array(selected)

# --- Crossover Simples (BLX-alpha) ---
def crossover(parent1, parent2, alpha=0.5):
    if np.random.rand() < CROSSOVER_RATE:
        cmin = np.minimum(parent1, parent2)
        cmax = np.maximum(parent1, parent2)
        diff = cmax - cmin
        low = cmin - alpha * diff
        high = cmax + alpha * diff
        child1 = np.random.uniform(low, high)
        child2 = np.random.uniform(low, high)
        return np.clip(child1, BOUNDS[0], BOUNDS[1]), np.clip(child2, BOUNDS[0], BOUNDS[1])
    else:
        return parent1.copy(), parent2.copy()

# --- Mutação Gaussiana ---
def mutate(ind):
    for i in range(DIM):
        if np.random.rand() < MUTATION_RATE:
            ind[i] += np.random.normal(0, 0.3)
            ind[i] = np.clip(ind[i], BOUNDS[0], BOUNDS[1])
    return ind

# --- Criação da Próxima Geração ---
def create_next_generation(pop, fitness):
    selected = tournament_selection(pop, fitness)
    next_gen = []
    for i in range(0, POP_SIZE, 2):
        p1, p2 = selected[i], selected[(i + 1) % POP_SIZE]
        c1, c2 = crossover(p1, p2)
        next_gen.append(mutate(c1))
        next_gen.append(mutate(c2))
    return np.array(next_gen[:POP_SIZE])

# --- Criação de Curvas de Nível da Função Alpine 2 ---
def plot_population(pop, gen, fitness_values, save_path=None):
    x = np.linspace(BOUNDS[0], BOUNDS[1], 200)
    y = np.linspace(BOUNDS[0], BOUNDS[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(X) * np.sin(X) * np.sqrt(Y) * np.sin(Y)

    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='f(x1, x2)')
    plt.scatter(pop[:, 0], pop[:, 1], c='red', s=40, label='População')
    plt.title(f'Geração {gen}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()

# --- Loop Principal ---
pop = initialize_population(POP_SIZE, DIM)
fitness_history = []
max_fit, mean_fit, min_fit = [], [], []

frames = []

os.makedirs("frames", exist_ok=True)

for gen in range(NUM_GENERATIONS):
    fitness = evaluate_population(pop)
    fitness_history.append(fitness)
    max_fit.append(np.max(fitness))
    mean_fit.append(np.mean(fitness))
    min_fit.append(np.min(fitness))

    # Salva frame do vídeo
    frame_path = f"frames/gen_{gen:03d}.png"
    plot_population(pop, gen, fitness, save_path=frame_path)
    frames.append(frame_path)

    # Cria próxima geração
    pop = create_next_generation(pop, fitness)

# --- Criação do Gráfico da Evolução do Fitness ---
plt.figure(figsize=(8, 5))
plt.plot(max_fit, label='Melhor Fitness', color='green')
plt.plot(mean_fit, label='Fitness Médio', color='blue')
plt.plot(min_fit, label='Pior Fitness', color='red')
plt.xlabel('Geração')
plt.ylabel('Fitness')
plt.title('Evolução do Fitness')
plt.legend()
plt.tight_layout()
plt.show()

# --- Geração do Vídeo ---
with imageio.get_writer('evolucao_populacao.mp4', fps=2.5) as writer:
    for filename in frames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("✅ Vídeo gerado: evolucao_populacao.mp4")

# --- Encontra o melhor indivíduo global ---
# Após a última geração
final_fitness = evaluate_population(pop)
best_index = np.argmax(final_fitness)
best_individual = pop[best_index]
best_fitness = final_fitness[best_index]

print("\n=== RESULTADO FINAL ===")
print(f"Melhor fitness: {best_fitness:.6f}")
print(f"x1 = {best_individual[0]:.6f}")
print(f"x2 = {best_individual[1]:.6f}")
