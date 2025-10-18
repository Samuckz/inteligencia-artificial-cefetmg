# =======================
# Algoritmo Clonalg para Alpine02 - Versão Única e Didática
# Autor: ChatGPT Data Analyst
# =======================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import time
import os

# Tente importar imageio, se não existir pede para instalar
try:
    import imageio
except ImportError:
    raise ImportError("Por favor, instale a biblioteca 'imageio' para a animação funcionar: pip install imageio")

# -------------------------------------------------------------------
# 1. Função Alpine02 (para duas dimensões)
def alpine02(x):
# x: vetor shape [N,2] ou [2,]
    return np.prod(np.sqrt(x) * np.sin(x), axis=-1)

# -------------------------------------------------------------------
# 2. Parâmetros do Algoritmo
NUM_DIM = 2
POP_SIZE = 30
N_SELECT = 10
N_CLONES = 5
N_ITER = 50
MUT_MIN = 0.02
MUT_MAX = 1.0
X_RANGE = (0, 10)
REPLACEMENT = 0.2 # Fração de renovação

# -------------------------------------------------------------------
# 3. Funções do Clonalg

def initialize_population(size, ndim, xrange):
    """Inicializa população [size,dim] nos limites dados."""
    return np.random.uniform(low=xrange[0], high=xrange[1], size=(size, ndim))

def evaluate_population(pop):
    """Avalia população usando a função Alpine02."""
    return alpine02(pop)

def select_best(pop, fitness, n_best):
    """Seleciona os n melhores indivíduos."""
    idx = np.argsort(fitness)[-n_best:]
    return pop[idx], fitness[idx]

def clone_population(pop, n_clones):
    """Replica cada indivíduo n_clones vezes."""
    return np.repeat(pop, n_clones, axis=0)

def mutate_population(clones, fitness, mut_min, mut_max, xrange):
    """Realiza mutação inversa à aptidão (melhores = perturbação menor)."""
    clones_mut = np.empty_like(clones)
    # Normaliza fitness
    fitness_norm = (fitness - np.min(fitness) + 1e-10) / (np.max(fitness) - np.min(fitness) + 1e-10)
    mutation_factors = mut_max - fitness_norm * (mut_max - mut_min)  # Melhores têm menos mutação
    for i in range(clones.shape[0]):
        sigma = mutation_factors[i % len(fitness)]
        clones_mut[i] = clones[i] + np.random.normal(0, sigma, clones.shape[1])
        clones_mut[i] = np.clip(clones_mut[i], xrange[0], xrange[1])
    return clones_mut

def replace_population(pop, fitness, new_individuals):
    """Substitui os piores indivíduos por novos randomicos."""
    idx = np.argsort(fitness)[:len(new_individuals)]
    pop[idx] = new_individuals
    return pop

# -------------------------------------------------------------------
# 4. CLONALG Principal
def clonalg(
pop_size, n_select, n_clones, ndim, xrange, mut_min, mut_max, n_iter, replacement_frac, verbose=True
):
    logs = []
    times = []
    pop = initialize_population(pop_size, ndim, xrange)
    history = []
    fit_history = []
    best_global = None
    best_fitness_global = -np.inf

    for it in range(n_iter):
        t0 = time.time()
        fitness = evaluate_population(pop)
        # Seleção dos melhores
        selected, selected_fitness = select_best(pop, fitness, n_select)
        # Clonagem
        clones = clone_population(selected, n_clones)
        # Mutação
        clones_mut = mutate_population(clones, selected_fitness, mut_min, mut_max, xrange)
        # Junção
        full_pop = np.vstack([pop, clones_mut])
        full_fitness = evaluate_population(full_pop)
        # Seleciona melhores para próxima geração
        idx_best = np.argsort(full_fitness)[-pop_size:]
        pop = full_pop[idx_best]
        fitness = full_fitness[idx_best]
        # Renovação de parte da população
        n_new = int(pop_size * replacement_frac)
        if n_new > 0:
            pop = replace_population(pop, fitness, initialize_population(n_new, ndim, xrange))
        # Tracking melhores
        best_idx = np.argmax(fitness)
        best = pop[best_idx]
        best_fitness = fitness[best_idx]
        if best_fitness > best_fitness_global:
            best_global = best.copy()
            best_fitness_global = best_fitness
        fit_history.append(best_fitness_global)
        history.append(pop.copy())
        t1 = time.time()
        times.append(t1 - t0)
        if verbose:
            print(f"Iteração {it+1:2d} | Melhor fitness: {best_fitness_global:.4f} | Tempo: {times[-1]:.3f}s")
    return best_global, best_fitness_global, history, fit_history, times

# -------------------------------------------------------------------
# 5. Visualização e Animação

def plot_surface_and_points(history, best_sol=None, xrange=X_RANGE, figsize=(10,8)):
    X = np.linspace(*xrange, 200)
    Y = np.linspace(*xrange, 200)
    Xg, Yg = np.meshgrid(X, Y)
    Z = alpine02(np.stack([Xg, Yg], axis=-1))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xg, Yg, Z, cmap='viridis', alpha=0.75, linewidth=0, antialiased=False)
    # Pontos geração final
    pops = history[-1]
    Zp = alpine02(pops)
    ax.scatter(pops[:,0], pops[:,1], Zp, color='red')
    if best_sol is not None:
        Zb = alpine02(best_sol)
        ax.scatter([best_sol[0]], [best_sol[1]], [Zb], color='gold', s=100, label='Melhor')
        ax.legend()
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    plt.title("Função Alpine02 e População Final")
    plt.tight_layout()
    plt.show()

def plot_convergence(fit_history):
    plt.figure(figsize=(8,4))
    plt.plot(fit_history, marker='o')
    plt.title("Convergência do CLONALG")
    plt.xlabel("Iteração")
    plt.ylabel("Melhor Fitness Encontrado")
    plt.grid()
    plt.tight_layout()
    plt.show()

def create_gif(history, filename, xrange=X_RANGE):
    frames = []
    X = np.linspace(*xrange, 200)
    Y = np.linspace(*xrange, 200)
    Xg, Yg = np.meshgrid(X, Y)
    Z = alpine02(np.stack([Xg, Yg], axis=-1))
    for k, pops in enumerate(history):
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Xg, Yg, Z, cmap='viridis', alpha=0.85, linewidth=0, antialiased=False)
        Zp = alpine02(pops)
        ax.scatter(pops[:,0], pops[:,1], Zp, color='red')
        ax.set_title(f"Evolução - Geração {k+1}")
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x)')
        # Ajusta o ângulo para todas as frames igual
        ax.view_init(elev=40, azim=45)
        plt.tight_layout()
        # Salva imagem em array numpy
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close()
    # Salva GIF
    imageio.mimsave(filename, frames, fps=5)
    print(f"Animação salva em: {filename}")

# -------------------------------------------------------------------
# 6. Relatório em Arquivo Texto
def gerar_relatorio(filename, pop_size, n_select, n_clones, n_iter, mut_min, mut_max, replacement, best_sol, best_fit, mean_time, total_time):
    with open(filename, "w") as f:
        f.write("# Relatório do Algoritmo Imunológico CLONALG para Função Alpine02\n\n")
        f.write("## Funcionamento do Algoritmo\n")
        f.write("O CLONALG simula o sistema imunológico adaptativo inspirando-se no funcionamento das células B e mecanismo de seleção clonal:\n\n")
        f.write("- Uma população de anticorpos (soluções) é mantida.\n")
        f.write("- Apenas os melhores anticorpos são clonados proporcionalmente à sua afinidade (fitness).\n")
        f.write("- Os clones sofrem mutação (hipermutação somática), sendo que melhores anticorpos sofrem mudanças menores.\n")
        f.write("- Após clonar e mutar, seleciona-se os melhores indivíduos para a próxima geração.\n")
        f.write("- Uma fração da população é renovada aleatoriamente, preservando diversidade e evitando convergência prematura (homeostase).\n\n")
        f.write("## Parâmetros Utilizados\n")
        f.write(f"- Tamanho População: {pop_size}\n")
        f.write(f"- Selecionados para clonar: {n_select}\n")
        f.write(f"- Clones por selecionado: {n_clones}\n")
        f.write(f"- Número de iterações: {n_iter}\n")
        f.write(f"- Mutação [mín, máx]: [{mut_min:.2f}, {mut_max:.2f}]\n")
        f.write(f"- Fração de renovação: {replacement:.2f}\n\n")
        f.write("## Funções e Métodos\n")
        f.write("- initialize_population: gera população inicial aleatória.\n")
        f.write("- evaluate_population: avalia a aptidão de cada solução.\n")
        f.write("- select_best: seleciona melhores soluções (maior afinidade).\n")
        f.write("- clone_population: realiza replicação clonal dos melhores.\n")
        f.write("- mutate_population: executa mutação inversamente proporcional ao fitness.\n")
        f.write("- replace_population: promove renovação (entrada de novos indivíduos).\n\n")
        f.write("## Caracterização Imunológica\n")
        f.write("- Efeito de seleção clonal: proliferação das soluções mais adaptadas.\n")
        f.write("- Hipermutação somática: geração de variantes dos clones.\n")
        f.write("- Homeostase: renovação parcial permanente da população.\n")
        f.write("- Memória imunológica: retenção das melhores soluções entre gerações.\n\n")
        f.write("## Resultados\n")
        f.write(f"- Melhor solução encontrada: {best_sol}\n")
        f.write(f"- Melhor valor da função: {best_fit:.6f}\n")
        f.write(f"- Tempo médio por geração: {mean_time:.4f}s\n")
        f.write(f"- Tempo total de execução: {total_time:.2f}s\n\n")
        f.write("## Observações Finais\n")
        f.write("O CLONALG é robusto para otimização multimodal e eficiente na busca de máximos/mínimos em funções desafiadoras, promovendo equilíbrio entre exploração e intensificação.\n")
    print(f"Relatório salvo em: {filename}")

# -------------------------------------------------------------------
# 7. EXECUÇÃO PRINCIPAL

if __name__ == '__main__':
    print("\n=== Execução do Algoritmo CLONALG para Alpine02 ===\n")
    ini_time = time.time()
    best_sol, best_fit, history, fit_history, exec_times = clonalg(
        pop_size=POP_SIZE, n_select=N_SELECT, n_clones=N_CLONES,
        ndim=NUM_DIM, xrange=X_RANGE,
        mut_min=MUT_MIN, mut_max=MUT_MAX,
        n_iter=N_ITER, replacement_frac=REPLACEMENT, verbose=True
    )
    fim_time = time.time()
    total_time = fim_time - ini_time
    mean_time = np.mean(exec_times)

print("\n--- RESULTADO FINAL ---")
print(f"Melhor solução: {best_sol}")
print(f"Melhor valor da função: {best_fit:.6f}")
print(f"Tempo total de execução: {total_time:.2f} s | Tempo médio por geração: {mean_time:.4f} s\n")

# Visualização
plot_surface_and_points(history, best_sol)
plot_convergence(fit_history)
gif_file = "clonalg_alpine02.gif"
create_gif(history, gif_file)

# Relatório
relatorio_file = "relatorio_clonalg_alpine02.txt"
gerar_relatorio(
    relatorio_file, POP_SIZE, N_SELECT, N_CLONES, N_ITER,
    MUT_MIN, MUT_MAX, REPLACEMENT, best_sol, best_fit, mean_time, total_time
)
print("\nExecução finalizada com sucesso. Consulte a pasta do script para visualizar gráficos, GIF e relatório.")