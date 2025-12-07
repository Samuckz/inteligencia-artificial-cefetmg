import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from statistics import mean, stdev

# ============================================================
# 1) Carregar base de dados
# ============================================================
df = pd.read_csv("heart.csv")

# Colunas categóricas da sua base
colunas_categoricas = [
    "Sex",            # M / F
    "ChestPainType",  # TA / ATA / NAP / ASY
    "RestingECG",     # Normal / ST / LVH
    "ExerciseAngina", # Y / N
    "ST_Slope"        # Up / Flat / Down
]

# ============================================================
# 2) Aplicar Label Encoding (adequado para árvores)
# ============================================================
df_encoded = df.copy()
label_encoders = {}

for col in colunas_categoricas:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separar X e y
X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]

# ============================================================
# 3) Função para avaliar modelos várias vezes
# ============================================================
def avalia_modelo(modelo, X, y, repeticoes=30):
    resultados = []
    for _ in range(repeticoes):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, shuffle=True
        )
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        resultados.append(accuracy_score(y_test, y_pred))
    return resultados

# ============================================================
# 4) Avaliação: Decision Tree
# ============================================================
tree_model = DecisionTreeClassifier()
tree_acc = avalia_modelo(tree_model, X, y)

print("=== RESULTADOS ÁRVORE DE DECISÃO ===")
print(f"Acurácia média: {mean(tree_acc):.4f}")
print(f"Desvio padrão: {stdev(tree_acc):.4f}")
print("--------------------------------------")

# ============================================================
# 5) Avaliação: Random Forest
# ============================================================
rf_model = RandomForestClassifier(n_estimators=200)
rf_acc = avalia_modelo(rf_model, X, y)

print("=== RESULTADOS RANDOM FOREST ===")
print(f"Acurácia média: {mean(rf_acc):.4f}")
print(f"Desvio padrão: {stdev(rf_acc):.4f}")
print("--------------------------------------")

# ============================================================
# 6) Análise: impacto da profundidade na Random Forest
# ============================================================
max_features = X.shape[1]
profundidades = range(1, max_features + 1)

print("\n=== VARIAÇÃO DA ACURÁCIA COM PROFUNDIDADE MÁXIMA ===")
for d in profundidades:
    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=d
    )
    accs = avalia_modelo(modelo, X, y, repeticoes=10)
    print(f"Profundidade {d}: Acurácia média = {mean(accs):.4f}")
