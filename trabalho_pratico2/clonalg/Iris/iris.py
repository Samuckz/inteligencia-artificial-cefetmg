import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# === Carregar base de dados ===
iris = load_iris()
X = iris.data
y = iris.target
n_features = 4
n_classes = 3

# Dividir em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=42
)

# === Parâmetros do algoritmo imunológico ===
n_antibodies = 30
n_generations = 100
clone_factor = 5
mutation_rate = 0.1
elite_fraction = 0.3

# === Inicialização ===
anticorpos = np.random.uniform(low=X_train.min(), high=X_train.max(), size=(n_antibodies, n_features))
labels = np.random.randint(0, n_classes, size=n_antibodies)

# === Funções auxiliares ===
def affinity(antibody, classe):
    X_c = X_train[y_train == classe]
    dist = np.mean(np.linalg.norm(X_c - antibody, axis=1))
    return 1 / (1 + dist)

def mutate(antibody, intensity=0.1):
    mutation = np.random.normal(0, intensity, size=antibody.shape)
    return antibody + mutation

# === Evolução ===
affinity_mean_history = []
affinity_max_history = []
affinity_min_history = []

for gen in range(n_generations):
    affinities = np.array([affinity(anticorpos[i], labels[i]) for i in range(n_antibodies)])
    
    # Guardar estatísticas
    affinity_mean_history.append(np.mean(affinities))
    affinity_max_history.append(np.max(affinities))
    affinity_min_history.append(np.min(affinities))
    
    # Selecionar elite
    n_elite = int(elite_fraction * n_antibodies)
    elite_idx = np.argsort(-affinities)[:n_elite]
    elite = anticorpos[elite_idx]
    elite_labels = labels[elite_idx]
    elite_aff = affinities[elite_idx]
    
    # Clonagem e mutação
    clones = []
    clone_labels = []
    for i in range(n_elite):
        num_clones = int(clone_factor * elite_aff[i] / elite_aff.max()) + 1
        for _ in range(num_clones):
            new_clone = mutate(elite[i], intensity=mutation_rate * (1 - elite_aff[i]))
            clones.append(new_clone)
            clone_labels.append(elite_labels[i])
    
    clones = np.array(clones)
    clone_labels = np.array(clone_labels)
    
    clone_aff = np.array([affinity(clones[i], clone_labels[i]) for i in range(len(clones))])
    
    # Combinar e selecionar os melhores
    all_anticorpos = np.vstack((anticorpos, clones))
    all_labels = np.hstack((labels, clone_labels))
    all_aff = np.hstack((affinities, clone_aff))
    
    best_idx = np.argsort(-all_aff)[:n_antibodies]
    anticorpos = all_anticorpos[best_idx]
    labels = all_labels[best_idx]

# === Obter centros de classe ===
class_centers = np.zeros((n_classes, n_features))
for c in range(n_classes):
    class_samples = anticorpos[labels == c]
    if len(class_samples) > 0:
        class_centers[c] = np.mean(class_samples, axis=0)
    else:
        class_centers[c] = np.mean(X_train[y_train == c], axis=0)

# === Classificação final ===
y_pred = []
for x in X_test:
    dist = np.linalg.norm(class_centers - x, axis=1)
    y_pred.append(np.argmin(dist))

accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia final: {accuracy * 100:.2f}%")

# === Gráfico 1: afinidade média, máxima e mínima ===
plt.figure(figsize=(8, 5))
plt.plot(affinity_mean_history, label='Afinidade média')
plt.plot(affinity_max_history, label='Afinidade máxima')
plt.plot(affinity_min_history, label='Afinidade mínima')
plt.title("Evolução da afinidade durante as gerações")
plt.xlabel("Geração")
plt.ylabel("Afinidade")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Gráfico 2: projeção 2D dos dados e anticorpos ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)
centers_pca = pca.transform(class_centers)

plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    plt.scatter(X_pca[y_train == i, 0], X_pca[y_train == i, 1],
                label=f"{iris.target_names[i]}", alpha=0.6, color=color)
    plt.scatter(centers_pca[i, 0], centers_pca[i, 1],
                color='black', edgecolor=color, s=200, marker='X')

plt.title("Projeção PCA das flores e vetores representativos (anticorpos finais)")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
