import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------
# Função Alpine 2 (maximização)
# ---------------------------------------------------
def alpine2(x):
    return np.prod(np.sqrt(x) * np.sin(x))

# ---------------------------------------------------
# Parâmetros do problema
# ---------------------------------------------------
n_dim = 2
bounds = [0, 10]

# ---------------------------------------------------
# Parâmetros do algoritmo Clonalg
# ---------------------------------------------------
pop_size = 50
num_generations = 100
num_clones = 10
num_random = 5
beta = 2.0

# ---------------------------------------------------
# Inicialização da população
# ---------------------------------------------------
pop = np.random.uniform(bounds[0], bounds[1], (pop_size, n_dim))
best_values = []
mean_values = []
worst_values = []
frames = []  # para o vídeo

# ---------------------------------------------------
# Loop de gerações
# ---------------------------------------------------
for gen in range(num_generations):
    fitness = np.array([alpine2(ind) for ind in pop])
    fit_norm = (fitness - np.min(fitness)) / (np.ptp(fitness) + 1e-12)

    clones = []
    for i in range(pop_size):
        n_clones_i = int(num_clones * (fit_norm[i] + 0.1))
        for _ in range(n_clones_i):
            clone = np.copy(pop[i])
            mutation_strength = np.exp(-beta * fit_norm[i])
            mutation = np.random.normal(0, mutation_strength, n_dim)
            clone += mutation
            clone = np.clip(clone, bounds[0], bounds[1])
            clones.append(clone)

    clones = np.array(clones)
    clone_fitness = np.array([alpine2(c) for c in clones])

    total_pop = np.vstack((pop, clones))
    total_fit = np.concatenate((fitness, clone_fitness))
    indices = np.argsort(-total_fit)
    pop = total_pop[indices[:pop_size]]

    # Introduz novos anticorpos aleatórios
    for i in range(num_random):
        pop[-(i + 1)] = np.random.uniform(bounds[0], bounds[1], n_dim)

    # Armazena estatísticas da geração
    new_fitness = np.array([alpine2(ind) for ind in pop])
    best_values.append(np.max(new_fitness))
    mean_values.append(np.mean(new_fitness))
    worst_values.append(np.min(new_fitness))

    # Guarda posição da população (para vídeo)
    frames.append(pop.copy())

# ---------------------------------------------------
# Resultado final
# ---------------------------------------------------
best_individual = pop[np.argmax([alpine2(ind) for ind in pop])]
best_fitness = alpine2(best_individual)

print(f"Melhor indivíduo encontrado: {best_individual}")
print(f"Melhor valor de fitness: {best_fitness:.4f}")

# ---------------------------------------------------
# Gráfico de convergência (melhor, médio e pior)
# ---------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(best_values, label='Melhor', color='blue')
plt.plot(mean_values, label='Médio', color='orange')
plt.plot(worst_values, label='Pior', color='red')
plt.title("Evolução dos Valores de Fitness (Clonalg - Alpine2)")
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# Geração do vídeo (mesmo formato do GA)
# ---------------------------------------------------
x = np.linspace(bounds[0], bounds[1], 200)
y = np.linspace(bounds[0], bounds[1], 200)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X) * np.sin(X) * np.sqrt(Y) * np.sin(Y)

fig, ax = plt.subplots(figsize=(6, 5))
contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
points = ax.scatter([], [], c='red', s=25)
ax.set_xlim(bounds[0], bounds[1])
ax.set_ylim(bounds[0], bounds[1])
ax.set_title("Evolução da População (Clonalg - Alpine2)")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")

def init():
    points.set_offsets(np.empty((0, 2)))  # <-- corrigido!
    return points,

def update(frame):
    data = frames[frame]
    points.set_offsets(data)
    ax.set_title(f"Geração {frame + 1}")
    return points,

ani = animation.FuncAnimation(
    fig, update, frames=len(frames),
    init_func=init, blit=True, interval=200, repeat=False
)

ani.save("clonalg_alpine2.mp4", writer="ffmpeg", fps=5)
print("Vídeo salvo como 'clonalg_alpine2.mp4'")
