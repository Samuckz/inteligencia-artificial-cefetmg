import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# --- Parâmetros do Algoritmo Genético ---
POP_SIZE = 50
NUM_GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
BOUNDS = [-100, 100]
DIM = 2  # n = 2

# --- Função Schaffer's F6 ---
def f_schaffer6(x):
    x1, x2 = x
    num = np.sin(np.sqrt(x1**2 + x2**2))**2 - 0.5
    den = (1 + 0.001 * (x1**2 + x2**2))**2
    return 0.5 - num / den

# --- Inicialização da População ---
def initialize_population(size, dim):
    return np.random.uniform(BOUNDS[0], BOUNDS[1], (size, dim))

# --- Avaliação da População ---
def evaluate_population(pop):
    return np.array([f_schaffer6(ind) for ind in pop])

# --- Seleção por Torneio ---
def tournament_selection(pop, fitness, k=3):
    selected = []
    for _ in range(len(pop)):
        aspirants_idx = np.random.randint(0, len(pop), k)
        best_idx = aspirants_idx[np.argmax(fitness[aspirants_idx])]
        selected.append(pop[best_idx])
    return np.array(selected)

# --- Crossover BLX-alpha ---
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
            ind[i] += np.random.normal(0, 5)
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

# --- Gráfico de Curvas de Nível + População ---
def plot_population(pop, gen, fitness_values, save_path=None):
    x = np.linspace(BOUNDS[0], BOUNDS[1], 400)
    y = np.linspace(BOUNDS[0], BOUNDS[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 - (np.sin(np.sqrt(X**2 + Y**2))**2 - 0.5) / (1 + 0.001 * (X**2 + Y**2))**2

    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='f(x, y)')
    plt.scatter(pop[:, 0], pop[:, 1], c='red', s=40, label='População')
    plt.title(f'Geração {gen}')
    plt.xlabel('x')
    plt.ylabel('y')
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
os.makedirs("frames_schaffer", exist_ok=True)

for gen in range(NUM_GENERATIONS):
    fitness = evaluate_population(pop)
    fitness_history.append(fitness)
    max_fit.append(np.max(fitness))
    mean_fit.append(np.mean(fitness))
    min_fit.append(np.min(fitness))

    # Salva frame do vídeo
    frame_path = f"frames_schaffer/gen_{gen:03d}.png"
    plot_population(pop, gen, fitness, save_path=frame_path)
    frames.append(frame_path)

    # Nova geração
    pop = create_next_generation(pop, fitness)

# --- Gráfico da Evolução do Fitness ---
plt.figure(figsize=(8, 5))
plt.plot(max_fit, label='Melhor Fitness', color='green')
plt.plot(mean_fit, label='Fitness Médio', color='blue')
plt.plot(min_fit, label='Pior Fitness', color='red')
plt.xlabel('Geração')
plt.ylabel('Fitness')
plt.title('Evolução do Fitness - Schaffer’s F6')
plt.legend()
plt.tight_layout()
plt.show()

# --- Geração do Vídeo ---
with imageio.get_writer('evolucao_schaffer.mp4', fps=10) as writer:
    for filename in frames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("✅ Vídeo gerado: evolucao_schaffer.mp4")
