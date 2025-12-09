import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import plot_tree

# Configuração para reproducibilidade
np.random.seed(42)

# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
print("=" * 60)
print("CARREGANDO E PREPARANDO OS DADOS")
print("=" * 60)

# Carregar dados
df = pd.read_csv('heart.csv')

print(f"\nShape dos dados: {df.shape}")
print(f"\nPrimeiras linhas:\n{df.head()}")
print(f"\nInformações sobre os dados:\n{df.info()}")
print(f"\nValores faltantes:\n{df.isnull().sum()}")

# Pré-processamento: codificar variáveis categóricas
label_encoders = {}
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separar features (X) e target (y)
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

print(f"\nNúmero de features: {X.shape[1]}")
print(f"Features: {list(X.columns)}")
print(f"\nDistribuição da variável target:\n{y.value_counts()}")

# 2. TREINAMENTO E AVALIAÇÃO - DECISION TREE
print("\n" + "=" * 60)
print("ÁRVORE DE DECISÃO (DECISION TREE)")
print("=" * 60)

dt_accuracies = []
n_executions = 30

for i in range(n_executions):
    # Dividir dados: 80% treino, 20% validação
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=i, stratify=y
    )
    
    # Treinar Decision Tree
    dt_classifier = DecisionTreeClassifier(max_depth=5)
    dt_classifier.fit(X_train, y_train)
    
    # Fazer predições
    y_pred = dt_classifier.predict(X_val)
    
    # Calcular acurácia
    accuracy = accuracy_score(y_val, y_pred)
    dt_accuracies.append(accuracy)
    
    if (i + 1) % 10 == 0:
        print(f"Execução {i + 1}/{n_executions} - Acurácia: {accuracy:.4f}")

# Estatísticas Decision Tree
dt_mean = np.mean(dt_accuracies)
dt_std = np.std(dt_accuracies)

print(f"\n{'Resultado Final - Decision Tree':^60}")
print(f"Acurácia Média: {dt_mean:.4f}")
print(f"Desvio Padrão: {dt_std:.4f}")
print(f"Acurácia Mínima: {np.min(dt_accuracies):.4f}")
print(f"Acurácia Máxima: {np.max(dt_accuracies):.4f}")

# 3. TREINAMENTO E AVALIAÇÃO - RANDOM FOREST
print("\n" + "=" * 60)
print("FLORESTA ALEATÓRIA (RANDOM FOREST)")
print("=" * 60)

rf_accuracies = []

for i in range(n_executions):
    # Dividir dados
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=i, stratify=y
    )
    
    # Treinar Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_classifier.fit(X_train, y_train)
    
    # Fazer predições
    y_pred = rf_classifier.predict(X_val)
    
    # Calcular acurácia
    accuracy = accuracy_score(y_val, y_pred)
    rf_accuracies.append(accuracy)
    
    if (i + 1) % 10 == 0:
        print(f"Execução {i + 1}/{n_executions} - Acurácia: {accuracy:.4f}")

# Estatísticas Random Forest
rf_mean = np.mean(rf_accuracies)
rf_std = np.std(rf_accuracies)

print(f"\n{'Resultado Final - Random Forest':^60}")
print(f"Acurácia Média: {rf_mean:.4f}")
print(f"Desvio Padrão: {rf_std:.4f}")
print(f"Acurácia Mínima: {np.min(rf_accuracies):.4f}")
print(f"Acurácia Máxima: {np.max(rf_accuracies):.4f}")

# 4. ANÁLISE: ACURÁCIA vs PROFUNDIDADE MÁXIMA (RANDOM FOREST)
print("\n" + "=" * 60)
print("ANÁLISE: ACURÁCIA vs PROFUNDIDADE MÁXIMA")
print("=" * 60)

n_features = X.shape[1]
max_depths = range(1, n_features + 1)
depth_accuracies_mean = []
depth_accuracies_std = []

print(f"\nAnalisando profundidades de 1 até {n_features}...")

for depth in max_depths:
    depth_accs = []
    
    for i in range(10):  # 10 execuções por profundidade
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y
        )
        
        rf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=depth, 
            random_state=i
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        depth_accs.append(accuracy_score(y_val, y_pred))
    
    mean_acc = np.mean(depth_accs)
    std_acc = np.std(depth_accs)
    depth_accuracies_mean.append(mean_acc)
    depth_accuracies_std.append(std_acc)
    
    print(f"Profundidade {depth:2d}: Acurácia = {mean_acc:.4f} (±{std_acc:.4f})")

# 5. VISUALIZAÇÕES
print("\n" + "=" * 60)
print("GERANDO VISUALIZAÇÕES")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Gráfico 1: Distribuição das Acurácias (Boxplot)
ax1 = axes[0, 0]
ax1.boxplot([dt_accuracies, rf_accuracies], labels=['Decision Tree', 'Random Forest'])
ax1.set_ylabel('Acurácia', fontsize=12)
ax1.set_title('Distribuição das Acurácias (30 execuções)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.7, 1.0])

# Gráfico 2: Acurácias ao longo das execuções
ax2 = axes[0, 1]
ax2.plot(range(1, n_executions + 1), dt_accuracies, 'o-', label='Decision Tree', alpha=0.7)
ax2.plot(range(1, n_executions + 1), rf_accuracies, 's-', label='Random Forest', alpha=0.7)
ax2.axhline(y=dt_mean, color='blue', linestyle='--', alpha=0.5, label=f'DT Média: {dt_mean:.4f}')
ax2.axhline(y=rf_mean, color='orange', linestyle='--', alpha=0.5, label=f'RF Média: {rf_mean:.4f}')
ax2.set_xlabel('Execução', fontsize=12)
ax2.set_ylabel('Acurácia', fontsize=12)
ax2.set_title('Acurácia por Execução', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Gráfico 3: Acurácia vs Profundidade Máxima
ax3 = axes[1, 0]
ax3.plot(max_depths, depth_accuracies_mean, 'o-', linewidth=2, markersize=8)
ax3.fill_between(
    max_depths, 
    np.array(depth_accuracies_mean) - np.array(depth_accuracies_std),
    np.array(depth_accuracies_mean) + np.array(depth_accuracies_std),
    alpha=0.3
)
ax3.set_xlabel('Profundidade Máxima', fontsize=12)
ax3.set_ylabel('Acurácia Média', fontsize=12)
ax3.set_title('Random Forest: Acurácia vs Profundidade Máxima', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(max_depths)

# Gráfico 4: Comparação Final
ax4 = axes[1, 1]
models = ['Decision Tree', 'Random Forest']
means = [dt_mean, rf_mean]
stds = [dt_std, rf_std]
colors = ['#3498db', '#e74c3c']

bars = ax4.bar(models, means, yerr=stds, capsize=10, color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Acurácia Média', fontsize=12)
ax4.set_title('Comparação Final dos Modelos', fontsize=14, fontweight='bold')
ax4.set_ylim([0.7, 1.0])
ax4.grid(True, alpha=0.3, axis='y')

# Adicionar valores nas barras
for bar, mean, std in zip(bars, means, stds):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{mean:.4f}\n±{std:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('heart_disease_analysis.png', dpi=300, bbox_inches='tight')
print("\nGráficos salvos em 'heart_disease_analysis.png'")
plt.show()

# 6. RESUMO FINAL
print("\n" + "=" * 60)
print("RESUMO FINAL")
print("=" * 60)
print(f"\nDecision Tree:")
print(f"  • Acurácia: {dt_mean:.4f} ± {dt_std:.4f}")
print(f"  • Meta: 0.83 ± 0.02")
print(f"  • Status: {'✓ ALCANÇADO' if abs(dt_mean - 0.83) <= 0.02 else '✗ NÃO ALCANÇADO'}")

print(f"\nRandom Forest:")
print(f"  • Acurácia: {rf_mean:.4f} ± {rf_std:.4f}")
print(f"  • Meta: 0.87 ± 0.02")
print(f"  • Status: {'✓ ALCANÇADO' if abs(rf_mean - 0.87) <= 0.02 else '✗ NÃO ALCANÇADO'}")

print(f"\nMelhor Profundidade (Random Forest): {max_depths[np.argmax(depth_accuracies_mean)]}")
print(f"Acurácia na melhor profundidade: {max(depth_accuracies_mean):.4f}")

# Após treinar uma árvore
plt.figure(figsize=(20,10))
plot_tree(dt_classifier, feature_names=X.columns, 
          class_names=['Sem Doença', 'Com Doença'], 
          filled=True, fontsize=10)
plt.show()

print("\n" + "=" * 60)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("=" * 60)