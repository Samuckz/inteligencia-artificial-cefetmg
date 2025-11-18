import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter

# --------------------------------------------------------
# Função Rastrigin
# --------------------------------------------------------
A = 10.0

def rastrigin(X):
    x = X[..., 0]
    y = X[..., 1]
    return A*2 + (x**2 - A*np.cos(2*np.pi*x)) + (y**2 - A*np.cos(2*np.pi*y))


# --------------------------------------------------------
# Classe PSO
# --------------------------------------------------------
class PSO:
    def __init__(self, func, bounds, n_particles=30, w=0.7, c1=1.5, c2=1.5, max_iter=100):
        self.func = func
        self.bounds = np.array(bounds)
        self.np = n_particles
        self.dim = 2
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter

    def reset(self):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        self.pos = np.random.uniform(low, high, size=(self.np, self.dim))

        vmax = (high - low) * 0.2
        self.vel = np.random.uniform(-vmax, vmax, size=(self.np, self.dim))

        self.pbest_pos = self.pos.copy()
        self.pbest_val = self.func(self.pos)

        g = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[g].copy()
        self.gbest_val = self.pbest_val[g]

        self.history_best = []
        self.history_pos = []

    def step(self):
        r1 = np.random.rand(self.np, self.dim)
        r2 = np.random.rand(self.np, self.dim)

        self.vel = (self.w * self.vel +
                    self.c1 * r1 * (self.pbest_pos - self.pos) +
                    self.c2 * r2 * (self.gbest_pos - self.pos))

        self.pos += self.vel

        low, high = self.bounds[:, 0], self.bounds[:, 1]
        self.pos = np.clip(self.pos, low, high)

        vals = self.func(self.pos)

        better = vals < self.pbest_val
        self.pbest_val[better] = vals[better]
        self.pbest_pos[better] = self.pos[better]

        g = np.argmin(self.pbest_val)
        if self.pbest_val[g] < self.gbest_val:
            self.gbest_val = self.pbest_val[g]
            self.gbest_pos = self.pbest_pos[g].copy()

    def run(self):
        self.reset()
        for _ in range(self.max_iter):
            self.step()
            self.history_best.append(self.gbest_val)
            self.history_pos.append(self.pos.copy())
        return self.gbest_pos, self.gbest_val


# --------------------------------------------------------
# Configurações gerais
# --------------------------------------------------------
bounds = np.array([[-5.12, 5.12], [-5.12, 5.12]])

param_list = [
    (0.5, 0.5),
    (1.0, 1.0),
    (1.5, 1.5),
    (2.0, 2.0),
    (2.5, 0.5),
    (0.5, 2.5)
]

results = {}


# --------------------------------------------------------
# Rodar PSO para vários valores de c1,c2
# --------------------------------------------------------
for c1, c2 in param_list:
    print(f"\nExecutando PSO com c1={c1}, c2={c2} ...")

    pso = PSO(
        func=rastrigin,
        bounds=bounds,
        n_particles=40,
        w=0.7,
        c1=c1,
        c2=c2,
        max_iter=120
    )

    best_pos, best_val = pso.run()
    results[(c1, c2)] = pso

    print(f"Melhor valor encontrado: {best_val:.6f}")


# --------------------------------------------------------
# Gerar gráficos de convergência
# --------------------------------------------------------
for (c1, c2), pso in results.items():
    plt.figure(figsize=(6, 4))
    plt.plot(pso.history_best, label=f"c1={c1}, c2={c2}")
    plt.xlabel("Iterações")
    plt.ylabel("Melhor valor encontrado")
    plt.title(f"Convergência do PSO (c1={c1}, c2={c2})")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"converg_c1_{c1}_c2_{c2}.png")
    plt.close()


# --------------------------------------------------------
# Criar GIF com movimento das partículas para um dos pares
# --------------------------------------------------------
c1, c2 = 1.5, 1.5    # escolha padrão
pso = results[(c1, c2)]

# Grade para contour
X = np.linspace(-5.12, 5.12, 300)
Y = np.linspace(-5.12, 5.12, 300)
XX, YY = np.meshgrid(X, Y)
ZZ = rastrigin(np.stack([XX, YY], axis=-1))

fig, ax = plt.subplots(figsize=(6, 6))
cs = ax.contour(XX, YY, ZZ, levels=40, cmap="viridis")
cbar = plt.colorbar(cs)
cbar.set_label("Valor da função Rastrigin")

scat = ax.scatter([], [], c="#FFFF00", edgecolors="black", s=60, linewidth=0.8)
title = ax.text(0.5, 1.03, "", transform=ax.transAxes, ha="center")

#ax.set_title(f"PSO (c1={c1}, c2={c2})")
ax.set_xlim(-5.12, 5.12)
ax.set_ylim(-5.12, 5.12)

def init():
    scat.set_offsets(np.empty((0, 2)))  # ← CORRETO
    title.set_text("")
    return scat, title

def update(frame):
    pts = pso.history_pos[frame]
    scat.set_offsets(pts)
    title.set_text(f"Iteração {frame+1}/{len(pso.history_pos)}")
    return scat, title

anim = animation.FuncAnimation(
    fig, update,
    frames=len(pso.history_pos),
    init_func=init,
    blit=True
)

anim.save("pso_rastrigin.gif", writer=PillowWriter(fps=10))
plt.close()

print("\nGIF salvo como pso_rastrigin.gif")
print("Gráficos de convergência salvos como converg_c1_X_c2_Y.png")
