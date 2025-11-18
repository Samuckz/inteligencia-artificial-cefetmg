import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================
# 1. Variáveis Linguísticas
# =============================

QueueSec = ctrl.Antecedent(np.arange(0, 51, 1), 'QueueSec')        # fila: 0 a 50 veículos
WaitTimeSec = ctrl.Antecedent(np.arange(0, 201, 1), 'WaitTimeSec')    # espera: 0 a 200 s
GreenAdj = ctrl.Consequent(np.arange(-10, 11, 1), 'GreenAdj')     # ajuste no tempo verde: -10 a +10 s

# =============================
# 2. Funções de Pertinência
# =============================

# QueueSec (fila secundária)
QueueSec['Small']  = fuzz.trimf(QueueSec.universe, [0, 0, 15])
QueueSec['Medium'] = fuzz.trimf(QueueSec.universe, [10, 25, 40])
QueueSec['Large']  = fuzz.trimf(QueueSec.universe, [30, 50, 50])

# WaitTimeSec (tempo de espera)
WaitTimeSec['Short']  = fuzz.trimf(WaitTimeSec.universe, [0, 0, 60])
WaitTimeSec['Medium'] = fuzz.trimf(WaitTimeSec.universe, [40, 100, 160])
WaitTimeSec['Long']   = fuzz.trimf(WaitTimeSec.universe, [120, 200, 200])

# GreenAdj (ajuste no tempo verde)
GreenAdj['Reduce']     = fuzz.trimf(GreenAdj.universe, [-10, -10, -2])
GreenAdj['NoChange']   = fuzz.trimf(GreenAdj.universe, [-3, 0, 3])
GreenAdj['Increase']   = fuzz.trimf(GreenAdj.universe, [2, 10, 10])

# =============================
# 3. Base de Regras Fuzzy
# =============================

rule1 = ctrl.Rule(QueueSec['Small'] & WaitTimeSec['Short'], GreenAdj['Reduce'])
rule2 = ctrl.Rule(QueueSec['Small'] & WaitTimeSec['Medium'], GreenAdj['NoChange'])
rule3 = ctrl.Rule(QueueSec['Medium'] & WaitTimeSec['Short'], GreenAdj['NoChange'])
rule4 = ctrl.Rule(QueueSec['Medium'] & WaitTimeSec['Long'], GreenAdj['Increase'])
rule5 = ctrl.Rule(QueueSec['Large'] & WaitTimeSec['Medium'], GreenAdj['Increase'])
rule6 = ctrl.Rule(QueueSec['Large'] & WaitTimeSec['Long'], GreenAdj['Increase'])
rule7 = ctrl.Rule(QueueSec['Medium'] & WaitTimeSec['Medium'], GreenAdj['NoChange'])
rule8 = ctrl.Rule(QueueSec['Large'] & WaitTimeSec['Short'], GreenAdj['NoChange'])
rule9 = ctrl.Rule(QueueSec['Small'] & WaitTimeSec['Long'], GreenAdj['Increase'])


# =============================
# 4. Sistema de Controle
# =============================

green_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
green_sim = ctrl.ControlSystemSimulation(green_ctrl)

# =============================
# 5. Superfície 3D
# =============================

queue_range = np.linspace(0, 50, 51)
wait_range = np.linspace(0, 200, 51)

X, Y = np.meshgrid(queue_range, wait_range)
Z = np.zeros_like(X)

for i in range(51):
    for j in range(51):
        sim = ctrl.ControlSystemSimulation(green_ctrl)  # novo a cada passo
        sim.input['QueueSec'] = X[i, j]
        sim.input['WaitTimeSec'] = Y[i, j]
        sim.compute()
        Z[i, j] = sim.output['GreenAdj']

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('Fila na Via Secundária (QueueSec)')
ax.set_ylabel('Tempo de Espera (WaitTimeSec)')
ax.set_zlabel('Ajuste no Verde (GreenAdj)')
ax.set_title('Superfície Fuzzy do Controlador de Tempo de Verde')
plt.show()
