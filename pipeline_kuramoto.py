# Pipeline de búsqueda y certificación (exportado de pipeline_kuramoto.ipynb; mismas celdas, mismo orden).

# %% [markdown]
# # Pipeline: cotas a la conectividad crítica del modelo de Kuramoto
#
# Este notebook busca **grafos densos con mínimos espurios** (equilibrios estables no sincronizados) del modelo de
# Kuramoto homogéneo y trata de **subir su conectividad** μ = (δ_min + 1)/n con el modelo de clusters. Al final
# certifica con aritmética de intervalos los mejores sistemas encontrados.
#
# | Paso | Qué hace | Sección |
# |---|---|---|
# | 1a | Dataset variado de grafos etiquetados (sincroniza / tiene mínimo espurio), equilibrado por familia | 2 |
# | 1b | Verificación de la GNN contra una línea de base | 3 |
# | 1c | Simulated annealing guiado por la GNN, desde varias familias | 4 |
# | 2 | Extraer θ* (mínimo espurio certificado numéricamente) | 5 |
# | 3 | Primer SDP con cada nodo = un cluster | 6 |
# | 3b | Semillas sintéticas (diez geometrías), GNN de sistemas de clusters y patadas a los twisted | 6b, 6c |
# | 3c | Rondas evolutivas sobre los mejores | 6d |
# | 4 | Selección de los grafos por pulido; CMA-ES + SDP + pulido convexo con el filtro TOPE | 7 |
# | 5 | Compresión a pocos clusters | 8 |
# | 6 | Certificación (simetría o intervalos) y blow-up explícito | 9 |
# | 7 | Certificación fina con margen chico (δ ≤ 10⁻⁶): el resultado final | 9b |
# | 8 | Figura final de evolución | 10 |
#
# **Fuentes de candidatos que entran al filtro TOPE:**
#
# - **Grafos** hallados por el annealing guiado por la GNN. Se eligen por μ después de un pulido corto (sección 7), no por
#   μ(G): grafos chicos con μ(G) ≈ 0.3 pueden caer en la cuenca de los mejores sistemas.
# - **Semillas sintéticas** de diez geometrías (`FAMILIAS_SEMILLAS`, `N_POR_FAMILIA` cada una). De cada familia entran las
#   `TOP_POR_FAMILIA` mejores por μ pre-pulido. `familias_semillas.csv` resume cuántas fueron factibles y hasta dónde
#   llegó cada familia.
# - **GNN de sistemas de clusters** ("pesada", sección 6c). Sus sistemas entran con la misma selección que las semillas.
# - **Patadas a los twisted:** todo θ* twisted (de grafos o de la GNN pesada) se reutiliza para fabricar semillas nuevas.
# - **Evolución** (sección 6d): `RONDAS_EVOLUTIVAS` rondas. Los padres son las `PADRES_EVOLUCION` mejores más la mejor
#   de cada geometría que no esté ya representada.
#
# **Reproducibilidad.** La sección 0b verifica con intervalos el template reportado (`datos/template_reportado.json`) en
# segundos y en cualquier máquina. La corrida completa (`RAPIDO = False`) lo vuelve a buscar desde cero: toda la
# aleatoriedad sale de `SEMILLA` y no hay corte por tiempo (`TIEMPO_MAX_HORAS = None`), así que en una misma máquina se
# repite igual. Con otro procesador o con otras versiones de las bibliotecas, la búsqueda puede tomar otro camino.
#
# **Cómo usarlo.** Correr las celdas en orden. Todos los parámetros que conviene tocar están en la sección 0.
# Con `RAPIDO = True` todo corre en pocos minutos (sirve para probar que funciona). Con `RAPIDO = False`
# se hace la corrida completa, que lleva unas horas.
#
# **Requisitos:** `pip install -r requirements.txt` (numpy, scipy, networkx, pandas, matplotlib, scikit-learn,
# torch, cvxpy, clarabel, cma, mpmath, joblib).
#
# **Carpeta.** El notebook espera estar en la raíz del repositorio, junto a la carpeta `datos/` (dataset y pesos de la
# GNN de partida, semillas de control de O'Keeffe). Todo lo que produce va a `resultados/`.
#

# %% [markdown]
# ## 0. Configuración
#
# **Parámetros importantes** (todos en esta celda):
#
# | Parámetro | Qué controla |
# |---|---|
# | `RAPIDO` | prueba rápida (True) o corrida completa (False) |
# | `N_JOBS` | núcleos en paralelo (−1 = todos) |
# | `N_MIN_SA`, `N_MAX_SA` | tamaño de los grafos del simulated annealing |
# | `UMBRAL_SDP1` | umbral de referencia del primer SDP (sólo se informa; la selección es por pulido) |
# | `DELTA_SDP1` | margen de estabilidad del primer SDP |
# | `DIAGONAL_UNO` | fija p_aa = 1 (conexión interna completa de cada cluster) en todos los SDP |
# | `N_TOP_DESCARTADOS` | cuántos descartados se dibujan en la figura final de evolución |
# | `TRATAR_TWISTED` | qué hacer con los twisted: "patada" (reutilizar), "descartar" o "nada" (tratarlos como cualquier candidato) |
# | `PATADAS_POR_TWISTED`, `TOP_PATADAS` | patadas por cada twisted y cuántas entran a TOPE |
# | `DELTA_INICIO_SEMILLAS` | nivel de δ en el que se evalúan y entran a TOPE las semillas |
# | `FACTOR_PRESELECCION`, `PASOS_PRESELECCION` | pre-pulido corto (pulido convexo) antes de elegir las que entran a TOPE |
# | `DELTAS` | escalera de márgenes del filtro TOPE |
# | `M_OBJETIVO` | meta del filtro TOPE (cota de O'Keeffe) |
# | `MARGEN_NIVEL0` | en el nivel 0 se descarta lo que quede más de esto por debajo del mejor |
# | `GENS_BASE`, `FACTOR_PRESUPUESTO` | generaciones de CMA-ES por nivel y cuánto crecen para el frente de Pareto |
# | `REINICIOS_SIGMA`, `K_GRANDE`, `GENS_K_GRANDE` | reinicios de CMA-ES por nivel y presupuesto reducido para k grande |
# | `PULIDO_SCP`, `ITERS_PULIDO`, `RADIO_PULIDO`, `RADIO_MIN_PULIDO` | pulido convexo secuencial al final de cada nivel |
# | `TAU_MAX`, `RATIO_HUECO`, `TOL_CAIDA` | compresión |
# | `Q_MAX`, `Q_MAX_BLOWUP`, `M_DENOMINADORES`, `M_MAX_BLOWUP`, `BLOWUP_MAX_ENTRADAS` | racionalización y certificación |
# | `FAMILIAS_SEMILLAS`, `N_POR_FAMILIA`, `TOP_POR_FAMILIA` | geometrías de semillas, cuántas por familia y cuántas de cada una entran a TOPE |
# | `PADRES_EVOLUCION`, `HIJOS_POR_PADRE`, `TOP_EVOLUCION`, `RONDAS_EVOLUTIVAS` | evolución (más la mejor de cada geometría) |
# | `PASOS_SELECCION_GRAFOS` | pasos del pulido con que se eligen los `MAX_CANDIDATOS` grafos |
# | `GNN_PESADA`, `N_SISTEMAS_PESADOS`, `TOP_PESADAS` | GNN de sistemas de clusters |
# | `CUPO_POR_FAMILIA`, `PESOS_POR_FAMILIA` | dataset equilibrado de la GNN de grafos |
# | `TIEMPO_MAX_HORAS` | límite opcional de tiempo (None = sin límite; con límite la corrida deja de ser reproducible) |
# | `N_CERT_FINA`, `DELTA_MAX_FINA`, `DPS_FINA`, `FACTORES_MARGEN_FINA` | certificación fina (sección 9b) |
#

# %%
import os
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")          # un hilo por proceso: el paralelismo va por joblib

import time, pickle, json, copy, warnings
T_INICIO = time.time()
from fractions import Fraction
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

RAPIDO = False
SEMILLA = 0
N_JOBS = -1

CARPETA = os.getcwd()
DATOS = os.path.join(CARPETA, "datos")
RES = os.path.join(CARPETA, "resultados")
os.makedirs(RES, exist_ok=True)

# --- Paso 1: dataset, GNN y simulated annealing ---
ARRANQUES = 48 if RAPIDO else 128          # arranques aleatorios del etiquetador
ITER_DESCENSO = 2000 if RAPIDO else 8000
N_NUEVAS = (10, 28)                         # tamaño de los grafos nuevos del dataset
GRAFOS_POR_FAMILIA_NUEVA = 12 if RAPIDO else 150
EPOCAS_GNN = 4 if RAPIDO else 120           # ajuste fino desde los pesos de partida
EPOCAS_GNN_FINAL = 2 if RAPIDO else 60      # reentrenamiento final con todo el dataset
N_MIN_SA, N_MAX_SA = 12, (16 if RAPIDO else 24)
CORRIDAS_POR_FAMILIA = 2 if RAPIDO else 8
PASOS_SA = 120 if RAPIDO else 1500
PESO_MU_SA = 0.3                            # peso de μ(G) en el puntaje (bajo: no exigimos densidad)
BETA_SA = 0.3                               # peso de log(probabilidad de la GNN)
TEMPERATURA_SA = 0.02
UMBRAL_GNN = 0.5                            # se llama al etiquetador si la GNN da más que esto
TOL_DIVERSIDAD = 0.02                       # dos θ* con firmas más cercanas que esto son "el mismo patrón"
MAX_CANDIDATOS = 8 if RAPIDO else 60           # se eligen por μ después de un pulido corto, no por μ(G)
PASOS_SELECCION_GRAFOS = 20 if RAPIDO else 150  # pasos del pulido de selección (en δ = DELTA_INICIO_SEMILLAS)

# --- dataset equilibrado para la GNN de grafos ---
EQUILIBRAR_DATASET = True
CUPO_POR_FAMILIA = 24 if RAPIDO else 300    # grafos por familia (tope)
FRACCION_POSITIVOS = 0.5                    # meta de positivos dentro de cada familia
INTENTOS_POR_FAMILIA = 2 if RAPIDO else 3   # se etiquetan a lo sumo INTENTOS × CUPO grafos nuevos por familia
PERTURBACIONES_MAX = 3                      # aristas que se cambian al perturbar un positivo
FRACCION_NEG_DIFICILES = 0.3                # parte de los negativos reservada a perturbados que sincronizan
FAMILIA_BLOQUES = True                      # agrega la familia "bloques" (modelo estocástico por bloques)
PESOS_POR_FAMILIA = True                    # cada celda (familia, clase) pesa lo mismo en la pérdida

# --- GNN de sistemas de clusters ("grafos pesados") ---
GNN_PESADA = True
N_SISTEMAS_PESADOS = 160 if RAPIDO else 4000   # tamaño total del dataset (repartido entre las familias)
K_PESADO = (4, 12)                          # cantidad de clusters de los sistemas del dataset
DELTA_ETIQUETA = 1e-4                       # margen pedido a D_a y λ₂ para llamar "positivo" a un sistema
DELTAS_SDP_DATASET = (2e-4, 3e-2)           # margen (log-uniforme) del SDP con que se arman los sistemas sdp_sintetica
ARRANQUES_PESADO = 24 if RAPIDO else 96     # arranques aleatorios del etiquetador de sistemas
ITER_DESCENSO_PESADO = 1500 if RAPIDO else 4000
EPOCAS_GNN_PESADA = 6 if RAPIDO else 150
CORRIDAS_SA_PESADO = 2 if RAPIDO else 16    # por familia de partida
PASOS_SA_PESADO = 40 if RAPIDO else 800
PESO_MU_SA_PESADO = 3.0                     # peso de μ del sistema en el puntaje (acá sí queremos μ alto)
TEMPERATURA_SA_PESADO = 0.01
PASO_P_SA, PASO_X_SA = 0.15, 0.1            # tamaño de los movimientos en P y (relativo) en x
VERIFICAR_CADA = 5                          # pasos mínimos entre dos llamadas al etiquetador en una corrida
TOP_PESADAS = 3 if RAPIDO else 20           # sistemas de la GNN pesada que entran a TOPE


# --- Paso 3: primer SDP ---
DELTA_SDP1 = 1e-3
UMBRAL_SDP1 = 0.4
DIAGONAL_UNO = True                         # p_aa = 1: cada cluster es un grafo completo

# --- Paso 4: CMA-ES + SDP con filtro TOPE ---
DELTAS = [1e-4, 1e-5] if RAPIDO else [1e-4, 1e-5, 1e-6, 1e-7]   # todos arrancan en 10⁻⁴
M_OBJETIVO = 0.69291                        # cota de O'Keeffe (arXiv:2610.21271)
COTA_CANALE = 11 / 16                       # 0.6875, cota de Canale (sólo para los gráficos)
N_TOP_DESCARTADOS = 5

# --- Paso 3b: semillas sintéticas y twisted ---
TRATAR_TWISTED = "patada"                   # "patada" | "descartar" | "nada"
TOL_TWISTED = 0.05                          # tolerancia relativa para decir que fases y tamaños son "twisted"
SEMILLAS_SINTETICAS = True
FAMILIAS_SEMILLAS = ["receta_general", "okeeffe", "cuatro_dos_partidos", "tres_grupos", "cinco_grupos",
                     "seis_siete", "centros_irregulares", "masas_desparejas", "tercios", "dos_escalas"]
N_POR_FAMILIA = 30 if RAPIDO else 2000      # semillas generadas por familia
TOP_POR_FAMILIA = 1 if RAPIDO else 15       # de cada familia entran a TOPE las mejores por μ pre-pulido
PATADAS_POR_TWISTED = 2 if RAPIDO else 4    # sistemas pateados por cada twisted
TOP_PATADAS = 4 if RAPIDO else 30          # patadas que entran a TOPE
DELTA_INICIO_SEMILLAS = 1e-4                # tiene que estar en DELTAS
GRUPOS_POSIBLES = [3, 4, 5, 6, 7]           # g: cantidad de grupos de fases
PROB_GRUPOS = [0.12, 0.33, 0.25, 0.18, 0.12]
JITTER_CENTROS = 0.15                       # corrimiento máximo de cada centro respecto de 2πj/g (rad)
EPS_PARTICION = (0.005, 0.2)                # separación ±ε entre las partes de un grupo partido (rad)
PROB_TERCIO = 0.2                           # probabilidad de partir un grupo en 3 en vez de 2
ALFA_DIRICHLET_RANGO = (2.0, 20.0)          # α ~ log-uniforme en este rango; α chico = tamaños más desparejos
FRACCION_OKEEFFE = 0.2                      # fracción de semillas con 4 grupos, todos partidos en mitades
K_MAX_SEMILLA = 14                          # se descartan semillas con más clusters que esto
PASOS_PRESELECCION = 20 if RAPIDO else 150  # la preselección usa el pulido convexo con estos pasos
PADRES_EVOLUCION = 3 if RAPIDO else 10      # padres por ronda (más la mejor de cada geometría no representada)
HIJOS_POR_PADRE = 4 if RAPIDO else 15       # mutaciones por padre
TOP_EVOLUCION = 3 if RAPIDO else 60         # hijos que entran a TOPE por ronda
RONDAS_EVOLUTIVAS = 1 if RAPIDO else 2      # rondas de evolución
MARGEN_TECHO_SEMILLAS = 0.002               # una semilla se descarta por techo sólo si techo < meta − esto
TIEMPO_MAX_HORAS = None                     # None = sin límite (reproducible). Con un número de horas, pasado ese
                                            # tiempo no se arrancan niveles ni rondas nuevas (el resultado pasa a
                                            # depender de la velocidad de la máquina)
FACTOR_PRESELECCION = 3                     # pasan al pre-pulido FACTOR × TOP semillas (por μ del SDP crudo)
GENS_PRESELECCION = 5 if RAPIDO else 15     # generaciones de CMA-ES del pre-pulido
MARGEN_NIVEL0 = 0.08
PULIDO_SCP = True                           # pulido convexo secuencial al final de cada nivel (y al comprimir)
ITERS_PULIDO = 40 if RAPIDO else 2000       # máximo de pasos del pulido por nivel (corta solo si no sube)
RADIO_PULIDO, RADIO_MIN_PULIDO = 0.02, 1e-7 # región de confianza inicial y mínima (rad; relativa en x)
# con el pulido, CMA-ES sólo tiene que elegir la cuenca: alcanza un presupuesto chico
GENS_BASE = 8 if RAPIDO else (60 if PULIDO_SCP else 300)            # por reinicio de CMA-ES
REINICIOS_SIGMA = (0.03,) if PULIDO_SCP else (0.03, 0.01, 0.003)    # cada nivel corre CMA-ES con estos σ
K_GRANDE = 14                               # con más clusters, una sola corrida de GENS_K_GRANDE generaciones
GENS_K_GRANDE = 8 if RAPIDO else 60
FACTOR_PRESUPUESTO = 2
SIGMA0, FACTOR_SIGMA, SIGMA_MIN = 0.05, 0.5, 1e-4
VENTANA_ESTANCAMIENTO, TOL_MEJORA = (15 if RAPIDO else (30 if PULIDO_SCP else 300)), 1e-6   # 300 = prácticamente apagado
X_MIN_FRAC = 0.2                            # tamaño mínimo de cada cluster: X_MIN_FRAC / k
R_MAX = 0.5                                 # |Σ x e^{iφ}| ≤ R_MAX (excluye el estado sincronizado)
RESCATE = True

# --- Paso 5: compresión ---
TAU_MAX = 1e-2                              # nunca se juntan fases separadas más que esto (rad)
RATIO_HUECO = 30                            # salto mínimo entre escalas de separación de fases
TOL_CAIDA = 1e-5                            # caída de μ tolerada al comprimir
GENS_PULIDO = 6 if RAPIDO else 30

# --- Paso 6: certificación ---
Q_MAX = 10000                               # denominador máximo para los tamaños x (x_a ≈ σ_a/Σσ, Σσ ≤ Q_MAX)
Q_MAX_BLOWUP = 120                          # además se prueban racionalizaciones con Σσ ≤ esto (armables como grafo)
OPCIONES_TAMANOS = 4                        # cuántas racionalizaciones de x se prueban (de cada tipo)
M_DENOMINADORES = [200, 1000, 5000, 20000, 10**5, 10**6, 10**7, 10**8]   # densidades p ≈ q/m
M_MAX_BLOWUP = 5000                         # hasta este m se arma el grafo explícito
BLOWUP_MAX_ENTRADAS = 2 * 10**7             # límite de memoria del blow-up: (Σσ)²·m
TOL_SIM = 1e-3                              # tolerancia para detectar simetría twisted
DPS = 50                                    # dígitos de precisión de mpmath
# --- Certificación fina (sección 9b) ---
DELTA_MAX_FINA = 1e-6                       # se certifican con este método los niveles con δ ≤ esto
N_CERT_FINA = 3 if RAPIDO else 10           # cuántos de los mejores de cada nivel se certifican
FACTORES_MARGEN_FINA = (1, 2, 4)            # margen del SDP re-resuelto: δ, 2δ, 4δ (el primero que certifica)
DPS_FINA = 80                               # dígitos de precisión
D_DEN_FINA = 2 * 10 ** 40                   # denominador común (par) de tamaños y densidades
RADIO_FINA = "1e-30"                        # radio de la caja de Krawczyk
TOL_FUSION_FINA = 1e-6                      # rad: clusters con fases más cercanas que esto se fusionan

rng = np.random.default_rng(SEMILLA)
REGISTRO = os.path.join(RES, "registro.txt")


def tiempo_agotado():
    """Ver `TIEMPO_MAX_HORAS`."""
    return TIEMPO_MAX_HORAS is not None and time.time() - T_INICIO > TIEMPO_MAX_HORAS * 3600


def log(*partes):
    """Imprime y guarda en resultados/registro.txt, con hora."""
    s = " ".join(str(p) for p in partes)
    print(s, flush=True)
    with open(REGISTRO, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + s + "\n")


def guardar(obj, nombre):
    with open(os.path.join(RES, nombre), "wb") as f:
        pickle.dump(obj, f)


log("=" * 70)
log(f"Inicio. RAPIDO={RAPIDO}  carpeta={CARPETA}")
for archivo in ("dataset_base.pkl", "gnn_base.pt", "control_okeeffe_k7_d1e-4.json"):
    assert os.path.exists(os.path.join(DATOS, archivo)), f"falta datos/{archivo}"
log("datos/ completo")

# %% [markdown]
# ## 0b. Verificación del resultado reportado
#
# Antes de buscar nada, esta celda verifica con aritmética de intervalos (`DPS_FINA` dígitos) el template reportado,
# `datos/template_reportado.json`: tamaños X_a/D y densidades Q_ab/D racionales y las fases aproximadas. Verifica que
# existe un equilibrio exacto (test de Krawczyk) y que es estrictamente estable en toda la caja (D_a > 0 y Cholesky con
# intervalos), y calcula μ(T) exacto. Tarda unos segundos.
#
# El template es el que produce la corrida completa de este notebook (`RAPIDO = False`, `SEMILLA = 0`) en la sección 9b,
# que lo guarda en `resultados/template_certificado.json`. Las funciones de certificación definidas acá son las que usa
# la sección 9b.
#

# %%
from fractions import Fraction
from mpmath import mp, iv


def fusionar_fases_iguales(x, phi, P, tol=TOL_FUSION_FINA):
    """Junta los clusters cuyas fases difieren menos de `tol` (tamaños sumados, densidades promedio)."""
    a = np.mod(phi, 2 * np.pi); o = np.argsort(a)
    grupos, actual = [], [o[0]]
    for i, j in zip(o[:-1], o[1:]):
        if a[j] - a[i] < tol:
            actual.append(j)
        else:
            grupos.append(actual); actual = [j]
    grupos.append(actual)
    if len(grupos) > 1 and a[grupos[0][0]] + 2 * np.pi - a[grupos[-1][-1]] < tol:
        grupos[0] = grupos.pop() + grupos[0]
    X = np.array([x[gr].sum() for gr in grupos])
    Ph = np.array([np.angle(np.sum(x[gr] * np.exp(1j * phi[gr]))) for gr in grupos])
    Pc = np.array([[np.sum(np.outer(x[gg], x[hh]) * P[np.ix_(gg, hh)]) / (X[a_] * X[b_]) for b_, hh in enumerate(grupos)]
                   for a_, gg in enumerate(grupos)])
    return X, Ph - Ph[0], Pc


def F_mp(x, P, phi):
    k = len(x)
    return [mp.fsum(P[a][b] * x[b] * mp.sin(phi[b] - phi[a]) for b in range(k) if b != a) for a in range(k)]


def corregir_equilibrio(x, P, phi):
    """Corrección de norma mínima en las densidades interiores para que F = 0 exacto (F es lineal en P)."""
    k = len(x)
    pares = [(a, b) for a in range(k) for b in range(a + 1, k) if 1e-9 < float(P[a][b]) < 1 - 1e-9]
    A = mp.matrix(k - 1, len(pares))
    for j, (a, b) in enumerate(pares):
        if a >= 1:
            A[a - 1, j] += x[b] * mp.sin(phi[b] - phi[a])
        if b >= 1:
            A[b - 1, j] += x[a] * mp.sin(phi[a] - phi[b])
    F = F_mp(x, P, phi)
    lam = mp.lu_solve(A * A.T, mp.matrix([-F[a] for a in range(1, k)]))
    dp = A.T * lam
    P2 = [fila[:] for fila in P]
    for j, (a, b) in enumerate(pares):
        P2[a][b] += dp[j]; P2[b][a] += dp[j]
    return P2, max(abs(dp[j]) for j in range(len(pares)))


def F_y_J_ctx(x, P, phi, ctx):
    """F_a (a = 2..k) y su jacobiano respecto de φ_2..φ_k, en mp o en iv."""
    k = len(x)
    F = [ctx.mpf(0)] * (k - 1); J = ctx.matrix(k - 1, k - 1)
    for a in range(1, k):
        for b in range(k):
            if b == a or P[a][b] == 0:
                continue
            w = P[a][b] * x[b]
            F[a - 1] += w * ctx.sin(phi[b] - phi[a])
            c = w * ctx.cos(phi[b] - phi[a])
            if b > 0:
                J[a - 1, b - 1] += c
            J[a - 1, a - 1] -= c
    return F, J


def cholesky_intervalos(A, n):
    """Cholesky de una matriz simétrica de intervalos: (éxito, mínimo pivote inferior)."""
    L = [[iv.mpf(0)] * n for _ in range(n)]
    piv_min = None
    for j in range(n):
        s = A[j][j] - sum((L[j][r] ** 2 for r in range(j)), iv.mpf(0))
        if s.a <= 0:
            return False, mp.mpf(s.a)
        piv_min = s.a if piv_min is None else min(piv_min, s.a)
        L[j][j] = iv.sqrt(s)
        for i in range(j + 1, n):
            L[i][j] = (A[i][j] - sum((L[i][r] * L[j][r] for r in range(j)), iv.mpf(0))) / L[j][j]
    return True, mp.mpf(piv_min)


def certificar_fino(x_f, phi_f, P_f):
    """Template racional (σ/D, Q/D) con equilibrio exacto (Krawczyk) y estabilidad estricta (Cholesky con intervalos)."""
    k = len(x_f)
    D = D_DEN_FINA
    X = [int(round(float(v) * D)) for v in x_f]
    X[int(np.argmax(x_f))] += D - sum(X)
    x = [mp.mpf(Xa) / D for Xa in X]
    phi = [mp.mpf(float(p)) for p in (phi_f - phi_f[0])]
    P = [[mp.mpf(0) if P_f[a, b] < 1e-12 else (mp.mpf(1) if P_f[a, b] > 1 - 1e-12 else mp.mpf(float(P_f[a, b])))
          for b in range(k)] for a in range(k)]
    for a in range(k):
        P[a][a] = mp.mpf(1)
    # 1. equilibrio exacto corrigiendo las densidades interiores
    P, corr = corregir_equilibrio(x, P, phi)
    if any(P[a][b] < 0 or P[a][b] > 1 for a in range(k) for b in range(k)):
        return dict(valido=False, motivo="la corrección sacó densidades de [0, 1]")
    # 2. densidades racionales con denominador par D (diagonal (D − 1)/D)
    Q = [[int(mp.nint(P[a][b] * D)) for b in range(k)] for a in range(k)]
    for a in range(k):
        Q[a][a] = D - 1
        for b in range(a + 1, k):
            Q[b][a] = Q[a][b]
    r = verificar_template(X, Q, D, phi)
    r["correccion"] = float(corr)
    return r


def verificar_template(X, Q, D, phi):
    """Verificación rigurosa de un template racional: tamaños X_a/D, densidades Q_ab/D (enteros, Σ X = D) y fases
    aproximadas phi (se refinan con Newton). Devuelve la validez y μ(T) exacto."""
    k = len(X)
    x = [mp.mpf(Xa) / D for Xa in X]
    phi = [mp.mpf(p) for p in phi]
    phi = [p - phi[0] for p in phi]
    Pq = [[mp.mpf(Q[a][b]) / D for b in range(k)] for a in range(k)]
    # 3. Newton en alta precisión + test de Krawczyk en una caja de radio RADIO_FINA
    for _ in range(80):
        Fv, J = F_y_J_ctx(x, Pq, phi, mp)
        paso = mp.lu_solve(J, mp.matrix(Fv))
        phi = [phi[0]] + [phi[a] - paso[a - 1] for a in range(1, k)]
        if mp.norm(paso) < mp.mpf(10) ** (-(DPS_FINA - 10)):
            break
    Fv, J = F_y_J_ctx(x, Pq, phi, mp)
    Y = mp.inverse(J)
    radio = mp.mpf(RADIO_FINA)
    caja = [iv.mpf([phi[a] - radio, phi[a] + radio]) for a in range(1, k)]
    xi = [iv.mpf(v) for v in x]; Pi = [[iv.mpf(v) for v in fila] for fila in Pq]
    F_pt, _ = F_y_J_ctx(xi, Pi, [iv.mpf(p) for p in phi], iv)
    _, J_iv = F_y_J_ctx(xi, Pi, [iv.mpf(phi[0])] + caja, iv)
    Y_iv = iv.matrix([[iv.mpf(Y[i, j]) for j in range(k - 1)] for i in range(k - 1)])
    Kr = iv.matrix([iv.mpf(phi[a]) for a in range(1, k)]) - Y_iv * iv.matrix(F_pt) + \
        (iv.eye(k - 1) - Y_iv * J_iv) * iv.matrix([caja[a] - iv.mpf(phi[a + 1]) for a in range(k - 1)])
    krawczyk = all(Kr[a].a > caja[a].a and Kr[a].b < caja[a].b for a in range(k - 1))
    # 4. estabilidad en toda la caja: D_a > 0 y M + v vᵀ ≻ 0 (v = √x), con M = X^{-1/2} L X^{-1/2}
    phc = [iv.mpf(phi[0])] + caja
    c_inf = [mp.mpf(sum((Pi[a][b] * xi[b] * iv.cos(phc[a] - phc[b]) for b in range(k)), iv.mpf(0)).a) for a in range(k)]
    sx = [iv.sqrt(v) for v in xi]
    A = [[iv.mpf(0)] * k for _ in range(k)]
    for a in range(k):
        for b in range(k):
            if a != b:
                w = Pi[a][b] * iv.cos(phc[a] - phc[b])
                A[a][b] = -sx[a] * sx[b] * w
                A[a][a] += xi[b] * w
    for a in range(k):
        for b in range(k):
            A[a][b] += sx[a] * sx[b]
    chol, piv = cholesky_intervalos(A, k)
    # 5. μ exacto del template
    mu = min(sum(Fraction(Q[a][b], D) * Fraction(X[b], D) for b in range(k)) for a in range(k))
    sep = min(abs(float(phi[a] - phi[b])) for a in range(k) for b in range(a + 1, k))
    valido = bool(krawczyk and chol and min(c_inf) > 0 and sep > 1e-6)
    return dict(valido=valido, mu=mu, krawczyk=krawczyk, cholesky=chol, pivote_min=float(piv),
                c_min=float(min(c_inf)), X=X, Q=Q, D=D, phi=[mp.nstr(p, 45) for p in phi])


# --- verificación del template reportado (datos/template_reportado.json) ---
with open(os.path.join(DATOS, "template_reportado.json")) as f:
    T_rep = json.load(f)
dps_antes = (mp.dps, iv.dps)
mp.dps = iv.dps = DPS_FINA
r_rep = verificar_template([int(v) for v in T_rep["X"]], [[int(v) for v in fila] for fila in T_rep["Q"]],
                           int(T_rep["D"]), T_rep["phi"])
mp.dps, iv.dps = dps_antes
assert r_rep["valido"] and str(r_rep["mu"]) == T_rep["mu"], "el template reportado no se verificó"
log(f"Template reportado ({len(T_rep['X'])} clusters): Krawczyk OK, Cholesky con intervalos OK "
    f"(pivote mínimo {r_rep['pivote_min']:.2e}), D_a ≥ {r_rep['c_min']:.4f}  →  μ_c ≥ μ(T) = {float(r_rep['mu']):.7f}")


# %% [markdown]
# ## 1. El etiquetador
#
# Busca mínimos espurios con muchos arranques (aleatorios, twisted dirigidos y, si se pasan, los de la GNN),
# los refina con Newton y los certifica numéricamente con el Hessiano: gradiente < 10⁻⁹ y λ₂ > 0 en el
# ortogonal de **1**. Devuelve el más robusto (mayor λ₂) y de qué tipo de semilla salió.
#
# - gradiente: $\partial E/\partial\theta_u = \sum_v A_{uv}\sin(\theta_u-\theta_v)$
# - Hessiano: $H = L(A\circ\cos\Delta\theta)$

# %%
def gradiente(A, th):
    c, s = np.cos(th), np.sin(th)
    return s * (c @ A) - c * (s @ A)


def hessiano(A, th):
    C = A * np.cos(th[:, None] - th[None, :])
    return np.diag(C.sum(1)) - C


def diferencias_en_aristas(A, th):
    u, v = np.nonzero(np.triu(A))
    return np.abs(np.angle(np.exp(1j * (th[u] - th[v]))))


def descenso(A, TH, iter_max, tol=1e-7):
    paso = 0.9 / A.sum(1).max()
    for _ in range(iter_max):
        G = gradiente(A, TH)
        if np.abs(G).max() < tol:
            break
        TH = TH - paso * G
    return TH


def newton(A, th, iters=40):
    gmax = np.inf
    for _ in range(iters):
        g = gradiente(A, th[None])[0]
        gmax = np.abs(g).max()
        if gmax < 1e-12:
            break
        th = th + np.linalg.lstsq(hessiano(A, th), -g, rcond=None)[0]
    return th, gmax


def analizar_estado(A, th):
    th, gmax = newton(A, th)
    autov = np.linalg.eigvalsh(hessiano(A, th))
    n = len(A)
    return dict(theta=th, gmax=gmax, lam2=autov[1],
                sincronizado=bool(diferencias_en_aristas(A, th).max() < 1e-6),
                estricto=bool(gmax < 1e-9 and autov[0] > -1e-8 and autov[1] > 1e-7 * n))


def semillas_dirigidas(A, qs=(1, 2, 3)):
    n = len(A)
    if n < 4:
        return np.zeros((0, n))
    _, V = np.linalg.eigh(np.diag(A.sum(1)) - A)
    return np.array([q * np.angle(V[:, k] + 1j * V[:, k + 1]) for k in (1, 2, 3) if k + 1 < n for q in qs])


def es_quasi_umbral(G):
    for comp in nx.connected_components(G):
        if len(comp) == 1:
            continue
        H = G.subgraph(comp)
        univ = [v for v in H if H.degree(v) == len(comp) - 1]
        if not univ or not es_quasi_umbral(H.subgraph(set(comp) - {univ[0]}).copy()):
            return False
    return True


def sincroniza_por_teorema(G):
    n = G.number_of_nodes()
    grados = [d for _, d in G.degree()]
    if nx.is_tree(G):
        return "árbol"
    if n == 4 and all(d == 2 for d in grados):
        return "ciclo C4"
    if min(grados) >= 0.75 * (n - 1):
        return "grado mínimo ≥ 3(n−1)/4"
    if es_quasi_umbral(G):
        return "quasi-umbral"
    return None


def etiquetar(G, rng, arranques=None, semillas_extra=None):
    arranques = arranques or ARRANQUES
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    n = len(A)
    teorema = sincroniza_por_teorema(G)
    bloques = [("aleatoria", rng.uniform(0, 2 * np.pi, (arranques, n))), ("twisted", semillas_dirigidas(A))]
    if semillas_extra is not None:
        bloques.append(("gnn", np.atleast_2d(semillas_extra)))
    origen = np.concatenate([[nombre] * len(b) for nombre, b in bloques])
    TH = descenso(A, np.vstack([b for _, b in bloques]), ITER_DESCENSO)
    mejor, origen_mejor, espurios_aleatorios = None, None, 0
    for i, th in enumerate(TH):
        if diferencias_en_aristas(A, th).max() < 1e-3:
            continue
        est = analizar_estado(A, th)
        if est["estricto"] and not est["sincronizado"]:
            espurios_aleatorios += origen[i] == "aleatoria"
            if mejor is None or est["lam2"] > mejor["lam2"]:
                mejor, origen_mejor = est, origen[i]
    if mejor is not None and teorema is not None:
        mejor = None                       # tolerancias numéricas: se descarta
    etiqueta = "no_sync_cert" if mejor is not None else ("sync_teo" if teorema else "sync_emp")
    return dict(etiqueta=etiqueta, teorema=teorema, theta=None if mejor is None else mejor["theta"],
                lam2=None if mejor is None else mejor["lam2"], cuenca=espurios_aleatorios / arranques,
                origen=origen_mejor)


def mu_grafo(G):
    return (min(d for _, d in G.degree()) + 1) / G.number_of_nodes()


t0 = time.time()
for nombre, G in [("C5", nx.cycle_graph(5)), ("K5", nx.complete_graph(5)),
                  ("C19(1..5)", nx.circulant_graph(19, range(1, 6)))]:
    r = etiquetar(G, rng)
    log(f"control etiquetador  {nombre:10s} → {r['etiqueta']:13s} origen={r['origen']}")
log(f"(esperado: C5 y C19(1..5) no_sync_cert, K5 sync_teo)  {time.time() - t0:.1f} s")

# %% [markdown]
# ## 2. Paso 1a: dataset con variedad y tamaño
#
# Se parte de un dataset previo (`datos/dataset_base.pkl`: todos los grafos conexos de hasta
# 7 vértices y ocho familias clásicas) y se le suman cuatro familias **densas y estructuradas**, que son las
# del régimen que nos interesa:
#
# - `er_denso`: Erdős–Rényi con p entre 0.45 y 0.85;
# - `circulante_denso`: anillos C_n(1..r) con r hasta 0.45·n, más saltos extra al azar;
# - `blowup_ciclo`: blow-ups de ciclos C_c (c = 4..8) con densidades al azar entre grupos vecinos y segundos vecinos;
# - `cayley`: grafos de Cayley de Z_a × Z_b y de diedrales, con conjunto generador al azar (cerrado por inversos).
#
# - `bloques`: modelo estocástico por bloques con 4 a 7 bloques de tamaños distintos, densidad interna alta y
#   densidades parciales al azar entre bloques. Es el análogo discreto del modelo de clusters.
#
# `cayley` y `blowup_ciclo` (junto con `producto` y `bloques_circulantes` del dataset base) se **reservan**
# para medir generalización fuera de familia.
#
# **Equilibrio (`EQUILIBRAR_DATASET`).** Cada familia se lleva a `CUPO_POR_FAMILIA` grafos, con la mitad
# positivos si se puede:
# 1. En las familias nuevas se generan grafos hasta llenar las dos mitades (o agotar `INTENTOS_POR_FAMILIA` × cupo).
# 2. En todas, si faltan positivos, se perturban positivos de la misma familia (1 a `PERTURBACIONES_MAX` aristas) y
#    se re-etiquetan. Los que siguen teniendo mínimo espurio son positivos nuevos; los que sincronizan son
#    **negativos difíciles** y ocupan hasta `FRACCION_NEG_DIFICILES` de la mitad negativa.
# 3. Si una familia no tiene positivos (árboles, grafos umbral: sincronizan por teorema) queda sólo con negativos; los
#    pesos por (familia, clase) del entrenamiento compensan el desbalance que quede.

# %%
def blowup_ciclo(c, rng, tam_max=5):
    tam = rng.integers(2, tam_max + 1, size=c)
    dens = {1: rng.uniform(0.5, 1.0), 2: rng.uniform(0.0, 0.5)}
    offs = np.concatenate([[0], np.cumsum(tam)])
    G = nx.empty_graph(int(offs[-1]))
    for a in range(c):
        for b in range(a, c):
            d = min(b - a, c - (b - a))
            for u in range(offs[a], offs[a + 1]):
                for v in range(offs[b], offs[b + 1]):
                    if u < v and (d == 0 or rng.random() < dens.get(d, 0.0)):
                        G.add_edge(u, v)
    return G


def cayley_aleatorio(rng, n_rango):
    while True:
        if rng.random() < 0.5:                                   # Z_a × Z_b
            a = int(rng.integers(2, 7)); b = int(rng.integers(2, 7))
            elems = [(i, j) for i in range(a) for j in range(b)]
            op = lambda g, h: ((g[0] + h[0]) % a, (g[1] + h[1]) % b)
            inv = lambda g: ((-g[0]) % a, (-g[1]) % b)
        else:                                                    # diedral D_m
            m = int(rng.integers(3, 15))
            elems = [(r, s) for s in (0, 1) for r in range(m)]
            op = lambda g, h: ((g[0] + (h[0] if g[1] == 0 else -h[0])) % m, g[1] ^ h[1])
            inv = lambda g: ((-g[0]) % m, 0) if g[1] == 0 else g
        N = len(elems)
        if not (n_rango[0] <= N <= n_rango[1]):
            continue
        idx = {g: i for i, g in enumerate(elems)}
        no_id = [g for g in elems if g != elems[0]]
        dens = rng.uniform(0.2, 0.8)
        S = set()
        for g in no_id:
            if rng.random() < dens:
                S |= {g, inv(g)}
        G = nx.empty_graph(N)
        G.add_edges_from((idx[g], idx[op(g, s)]) for g in elems for s in S)
        if nx.is_connected(G):
            return G


def generar_nueva(familia, rng):
    n = int(rng.integers(*N_NUEVAS))
    if familia == "er_denso":
        return nx.gnp_random_graph(n, rng.uniform(0.45, 0.85), seed=int(rng.integers(1e9)))
    if familia == "circulante_denso":
        r = int(rng.integers(1, max(2, int(0.45 * n))))
        extra = rng.integers(1, n // 2 + 1, size=int(rng.integers(0, 3))).tolist()
        return nx.circulant_graph(n, sorted(set(range(1, r + 1)) | set(extra)))
    if familia == "blowup_ciclo":
        return blowup_ciclo(int(rng.integers(4, 9)), rng)
    if familia == "cayley":
        return cayley_aleatorio(rng, N_NUEVAS)
    if familia == "bloques":
        b = int(rng.integers(4, 8))
        tam = np.maximum(1, np.round(rng.dirichlet(np.full(b, 2.0)) * n)).astype(int)
        dens = np.triu(rng.uniform(0, 1, (b, b)) ** rng.uniform(0.5, 2), 1)
        dens = dens + dens.T
        np.fill_diagonal(dens, rng.uniform(0.7, 1.0, b))
        return nx.convert_node_labels_to_integers(
            nx.stochastic_block_model(tam.tolist(), dens.tolist(), seed=int(rng.integers(1e9))))
    raise ValueError(familia)


FAMILIAS_NUEVAS = ["er_denso", "circulante_denso", "blowup_ciclo", "cayley"] + (["bloques"] if FAMILIA_BLOQUES else [])
FAMILIAS_FUERA = ["cayley", "blowup_ciclo", "producto", "bloques_circulantes"]

with open(os.path.join(DATOS, "dataset_base.pkl"), "rb") as f:
    dataset = pickle.load(f)
for r in dataset:
    r.setdefault("origen", None)
log(f"dataset base: {len(dataset)} grafos")


def registro(G, familia, semilla):
    warnings.filterwarnings("ignore")          # los procesos de joblib no heredan el filtro
    G = nx.convert_node_labels_to_integers(G)
    return dict(G=G, familia=familia, n=G.number_of_nodes(),
                **etiquetar(G, np.random.default_rng(semilla)))


from joblib import Parallel, delayed

def es_positivo(r):
    return r["etiqueta"] == "no_sync_cert"


def etiquetar_tanda(grafos, familia, procedencia):
    salida = Parallel(n_jobs=N_JOBS)(delayed(registro)(G, familia, int(rng.integers(1e9))) for G in grafos)
    for r in salida:
        r["procedencia"] = procedencia
    return salida


def perturbar(G, rng_l):
    """Agrega o saca entre 1 y PERTURBACIONES_MAX aristas al azar, manteniendo el grafo conexo."""
    n = G.number_of_nodes()
    for _ in range(50):
        H = G.copy()
        for _ in range(int(rng_l.integers(1, PERTURBACIONES_MAX + 1))):
            u, v = rng_l.choice(n, 2, replace=False)
            H.remove_edge(u, v) if H.has_edge(u, v) else H.add_edge(u, v)
        if nx.is_connected(H):
            return H
    return None


def equilibrar_familia(familia, regs, generar):
    """Lleva una familia a CUPO_POR_FAMILIA grafos con FRACCION_POSITIVOS de positivos, si se puede."""
    n_pos = int(round(CUPO_POR_FAMILIA * FRACCION_POSITIVOS)); n_neg = CUPO_POR_FAMILIA - n_pos
    n_dif = int(FRACCION_NEG_DIFICILES * n_neg)
    for r in regs:
        r.setdefault("procedencia", "base")
    pos = [r for r in regs if es_positivo(r)]
    neg = [r for r in regs if not es_positivo(r)]
    vistos = {nx.weisfeiler_lehman_graph_hash(r["G"]) for r in regs}
    presupuesto = INTENTOS_POR_FAMILIA * CUPO_POR_FAMILIA
    tanda_max = max(16, min(64, CUPO_POR_FAMILIA))
    # 1. grafos nuevos de la familia
    while generar and (len(pos) < n_pos or len(neg) < n_neg) and presupuesto > 0:
        tanda = []
        for _ in range(20 * tanda_max):
            if len(tanda) >= min(tanda_max, presupuesto):
                break
            G = generar_nueva(familia, rng)
            h = nx.weisfeiler_lehman_graph_hash(G)
            if nx.is_connected(G) and h not in vistos:
                vistos.add(h); tanda.append(G)
        if not tanda:
            break
        presupuesto -= len(tanda)
        for r in etiquetar_tanda(tanda, familia, "generado"):
            (pos if es_positivo(r) else neg).append(r)
    # 2. perturbaciones de positivos: positivos fabricados y negativos difíciles
    dificiles = []
    presupuesto = INTENTOS_POR_FAMILIA * CUPO_POR_FAMILIA
    while pos and (len(pos) < n_pos or len(dificiles) < n_dif) and presupuesto > 0:
        tanda = []
        for _ in range(20 * tanda_max):
            if len(tanda) >= min(tanda_max, presupuesto):
                break
            H = perturbar(pos[int(rng.integers(len(pos)))]["G"], rng)
            if H is None:
                continue
            h = nx.weisfeiler_lehman_graph_hash(H)
            if h not in vistos:
                vistos.add(h); tanda.append(H)
        if not tanda:
            break
        presupuesto -= len(tanda)
        for r in etiquetar_tanda(tanda, familia, "perturbado"):
            if es_positivo(r):
                pos.append(r)
            else:
                r["procedencia"] = "negativo_dificil"; dificiles.append(r)
    rng.shuffle(pos); rng.shuffle(neg)
    dificiles = dificiles[:n_dif]
    return pos[:n_pos] + dificiles + neg[:n_neg - len(dificiles)]


t0 = time.time()
if EQUILIBRAR_DATASET:
    por_familia = {}
    for r in dataset:
        por_familia.setdefault(r["familia"], []).append(r)
    for familia in FAMILIAS_NUEVAS:
        por_familia.setdefault(familia, [])
    dataset = []
    for familia, regs in por_familia.items():
        elegidos = equilibrar_familia(familia, regs, familia in FAMILIAS_NUEVAS)
        dataset += elegidos
        pos = sum(es_positivo(r) for r in elegidos)
        log(f"  {familia:20s} {len(elegidos):4d} grafos, {pos:4d} positivos "
            f"({sum(r['procedencia'] == 'perturbado' for r in elegidos)} perturbados, "
            f"{sum(r['procedencia'] == 'negativo_dificil' for r in elegidos)} negativos difíciles)  ({time.time() - t0:.0f} s)")
else:
    for familia in FAMILIAS_NUEVAS:
        grafos = []
        while len(grafos) < GRAFOS_POR_FAMILIA_NUEVA:
            G = generar_nueva(familia, rng)
            if nx.is_connected(G):
                grafos.append(G)
        nuevos = etiquetar_tanda(grafos, familia, "generado")
        dataset += nuevos
        pos = sum(es_positivo(r) for r in nuevos)
        log(f"  {familia:18s} {len(nuevos)} grafos, {pos} con mínimo espurio ({time.time() - t0:.0f} s)")

tabla = pd.DataFrame([{k: r[k] for k in ("familia", "n", "etiqueta")} for r in dataset])
print(pd.crosstab(tabla.familia, tabla.etiqueta, margins=True))
guardar(dataset, "dataset_pipeline.pkl")
log(f"dataset total: {len(dataset)} grafos → resultados/dataset_pipeline.pkl")

# %% [markdown]
# ## 3. Paso 1b: ¿la GNN predice bien?
#
# Misma arquitectura que en `ml_minimos_espurios/` (se cargan sus pesos y se hace **ajuste fino** con el
# dataset nuevo). Se compara contra una **línea de base** (gradient boosting sobre rasgos clásicos del grafo).
#
# **Parámetros de calidad:**
# - **AUC** (área bajo la curva ROC): 0.5 es azar, 1 es perfecto. Es la métrica principal.
# - **recall**: fracción de grafos con mínimo espurio que la red detecta (lo que más importa como filtro).
# - **precisión**: fracción de los que la red marca que realmente tienen mínimo espurio.
# - **tasa de certificación con la fase de la GNN**: fracción de positivos en los que *un solo* arranque
#   desde la fase propuesta por la GNN encuentra el mínimo, comparada con la cuenca (probabilidad de lograrlo
#   con un arranque aleatorio).
#
# La GNN sirve como filtro si su AUC fuera de familia es claramente mayor que 0.5 y no mucho peor que la
# línea de base.

# %%
import torch
import torch.nn as nn
torch.set_num_threads(max(1, os.cpu_count() or 1))
torch.manual_seed(SEMILLA)
K_PE, K_AZAR = 8, 4


def etiqueta_binaria(r):
    return int(r["etiqueta"] == "no_sync_cert")


def entrada_gnn(G, rng_local):
    A = nx.to_numpy_array(G, nodelist=range(G.number_of_nodes()))
    n = len(A)
    grados = A.sum(1)
    Dm = 1 / np.sqrt(grados)
    _, V = np.linalg.eigh(np.eye(n) - Dm[:, None] * A * Dm[None, :])
    pe = np.zeros((n, K_PE))
    k = min(K_PE, n - 1)
    pe[:, :k] = V[:, 1:k + 1] * rng_local.choice([-1, 1], size=k)
    rasgos_v = np.hstack([grados[:, None] / n, grados[:, None] / grados.max(), pe,
                          rng_local.normal(size=(n, K_AZAR))])
    return A / grados[:, None], rasgos_v


def lote(regs, rng_local, con_fases=True):
    n_max = max(r["n"] for r in regs)
    B = len(regs)
    A = np.zeros((B, n_max, n_max)); Xv = np.zeros((B, n_max, 2 + K_PE + K_AZAR))
    M = np.zeros((B, n_max)); TH = np.zeros((B, n_max)); tiene = np.zeros(B)
    for b, r in enumerate(regs):
        a, xv = entrada_gnn(r["G"], rng_local)
        n = r["n"]
        A[b, :n, :n], Xv[b, :n], M[b, :n] = a, xv, 1
        if con_fases and r.get("theta") is not None:
            TH[b, :n], tiene[b] = r["theta"], 1
    t = lambda z: torch.tensor(z, dtype=torch.float32)
    return t(A), t(Xv), t(M), t(TH), t(tiene), t([etiqueta_binaria(r) if "etiqueta" in r else 0 for r in regs])


class GNNKuramoto(nn.Module):
    def __init__(self, d_in=2 + K_PE + K_AZAR, d=64, capas=4):
        super().__init__()
        self.entrada = nn.Linear(d_in, d)
        self.propio = nn.ModuleList([nn.Linear(d, d) for _ in range(capas)])
        self.vecinos = nn.ModuleList([nn.Linear(d, d) for _ in range(capas)])
        self.norma = nn.ModuleList([nn.LayerNorm(d) for _ in range(capas)])
        self.cabeza_grafo = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.cabeza_fase = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2))

    def forward(self, A, X, M):
        h = torch.relu(self.entrada(X)) * M[..., None]
        for propio, vecinos, norma in zip(self.propio, self.vecinos, self.norma):
            h = (h + torch.relu(norma(propio(h) + vecinos(A @ h)))) * M[..., None]
        promedio = h.sum(1) / M.sum(1, keepdim=True)
        maximo = (h - 1e9 * (1 - M[..., None])).max(1).values
        logit = self.cabeza_grafo(torch.cat([promedio, maximo], -1)).squeeze(-1)
        z = self.cabeza_fase(h)
        return logit, torch.atan2(z[..., 1], z[..., 0])


def perdida_fase(fase, TH, M, tiene):
    n = M.sum(1)
    def coh(signo):
        d = fase - signo * TH
        return torch.sqrt(((torch.cos(d) * M).sum(1) / n) ** 2 + ((torch.sin(d) * M).sum(1) / n) ** 2 + 1e-12)
    return ((1 - torch.maximum(coh(1), coh(-1))) * tiene).sum() / tiene.sum().clamp(min=1)


def pesos_muestras(regs, idx):
    """Con `PESOS_POR_FAMILIA`, cada celda (familia, clase) suma el mismo peso; si no, balance global de clases."""
    idx = np.asarray(idx)
    yv = np.array([etiqueta_binaria(regs[i]) for i in idx])
    if PESOS_POR_FAMILIA:
        celdas = [(regs[i]["familia"], int(v)) for i, v in zip(idx, yv)]
        cuenta = pd.Series(celdas).value_counts()
        w = np.array([1.0 / cuenta[c] for c in celdas])
    else:
        pos = max(1, yv.sum())
        w = np.where(yv == 1, (len(yv) - pos) / pos, 1.0)
    return dict(zip(idx.tolist(), (w * len(w) / w.sum()).tolist()))


def entrenar(modelo, idx, epocas, lr=5e-4, tam_lote=32, peso_fase=0.5, datos=None, armar_lote=None):
    datos = dataset if datos is None else datos
    armar_lote = lote if armar_lote is None else armar_lote
    rng_local = np.random.default_rng(SEMILLA)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=1e-5)
    pesos = pesos_muestras(datos, idx)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    historia = []
    for ep in range(epocas):
        modelo.train()
        total = 0.0
        for orden in np.array_split(rng_local.permutation(idx), max(1, len(idx) // tam_lote)):
            A, Xv, M, TH, tiene, yb = armar_lote([datos[j] for j in orden], rng_local)
            wb = torch.tensor([pesos[int(j)] for j in orden], dtype=torch.float32)
            logit, fase = modelo(A, Xv, M)
            perdida = (bce(logit, yb) * wb).mean() + peso_fase * perdida_fase(fase, TH, M, tiene)
            opt.zero_grad(); perdida.backward(); opt.step()
            total += perdida.item() * len(yb)
        historia.append(total / len(idx))
        if ep % max(1, epocas // 6) == 0 or ep == epocas - 1:
            log(f"  época {ep:4d}  pérdida {historia[-1]:.4f}")
    return historia


@torch.no_grad()
def predecir(modelo, regs, semilla=1, armar_lote=None):
    armar_lote = lote if armar_lote is None else armar_lote
    modelo.eval()
    regs = [r if isinstance(r, dict) else dict(G=r, n=r.number_of_nodes()) for r in regs]
    rng_local = np.random.default_rng(semilla)
    probs, fases = [], []
    for i in range(0, len(regs), 64):
        A, Xv, M, *_ = armar_lote(regs[i:i + 64], rng_local, con_fases=False)
        logit, fase = modelo(A, Xv, M)
        probs += torch.sigmoid(logit).tolist()
        fases += [f[:int(m.sum())].numpy() for f, m in zip(fase, M)]
    return np.array(probs), fases


# partición: dentro de familia (75/25) y fuera de familia
y = np.array([etiqueta_binaria(r) for r in dataset])
fuera = np.array([r["familia"] in FAMILIAS_FUERA for r in dataset])
dentro = np.flatnonzero(~fuera)
rng.shuffle(dentro)
corte = int(0.75 * len(dentro))
idx_entrena, idx_prueba, idx_fuera = dentro[:corte], dentro[corte:], np.flatnonzero(fuera)
log(f"entrenamiento {len(idx_entrena)}, prueba {len(idx_prueba)}, fuera de familia {len(idx_fuera)}")
log(f"fracción de positivos: {y[idx_entrena].mean():.2f} / {y[idx_prueba].mean():.2f} / {y[idx_fuera].mean():.2f}")

gnn = GNNKuramoto()
gnn.load_state_dict(torch.load(os.path.join(DATOS, "gnn_base.pt")))
t0 = time.time()
historia = entrenar(gnn, idx_entrena, EPOCAS_GNN)
log(f"ajuste fino de la GNN: {time.time() - t0:.0f} s")

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve


def rasgos(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    grados = np.array([d for _, d in G.degree()])
    esp = np.sort(nx.normalized_laplacian_spectrum(G))
    cuant = np.interp(np.linspace(0, 1, 16), np.linspace(0, 1, n), esp)
    girth = nx.girth(G)
    return np.concatenate([[n, m / (n * (n - 1) / 2), (grados.min() + 1) / n, grados.mean() / n,
                            grados.std() / n, m - n + 1, sum(nx.triangles(G).values()) / 3 / n,
                            0 if np.isinf(girth) else girth, nx.average_clustering(G), nx.diameter(G),
                            esp[1]], cuant])


def metricas(y_true, prob, umbral=0.5):
    pred = (prob >= umbral).astype(int)
    auc = roc_auc_score(y_true, prob) if 0 < y_true.sum() < len(y_true) else float("nan")
    return dict(AUC=auc, precision=precision_score(y_true, pred, zero_division=0),
                recall=recall_score(y_true, pred, zero_division=0))


X = np.array([rasgos(r["G"]) for r in dataset])
base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEMILLA,
                                      class_weight="balanced")
base.fit(X[idx_entrena], y[idx_entrena])
prob_base = base.predict_proba(X)[:, 1]
prob_gnn, fases_gnn = predecir(gnn, dataset)

filas = {}
for modelo, prob in [("línea de base", prob_base), ("GNN", prob_gnn)]:
    for parte, idx in [("prueba", idx_prueba), ("fuera de familia", idx_fuera)]:
        filas[(modelo, parte)] = metricas(y[idx], prob[idx])
tabla_metricas = pd.DataFrame(filas).T.round(3)
log("\n" + tabla_metricas.to_string())


def metricas_por_familia(regs, idx, prob, nombre):
    """AUC y recall por familia (AUC sólo si la familia tiene positivos y negativos en esa parte)."""
    yv = np.array([etiqueta_binaria(regs[i]) for i in idx])
    fam = np.array([regs[i]["familia"] for i in idx])
    filas_f = []
    for f_ in sorted(set(fam)):
        m_ = fam == f_
        mt = metricas(yv[m_], prob[np.asarray(idx)][m_])
        filas_f.append(dict(modelo=nombre, familia=f_, grafos=int(m_.sum()), positivos=int(yv[m_].sum()),
                            AUC=round(mt["AUC"], 3), recall=round(mt["recall"], 3), precision=round(mt["precision"], 3)))
    return filas_f


idx_eval = np.concatenate([idx_prueba, idx_fuera])
tabla_familias = pd.DataFrame(metricas_por_familia(dataset, idx_eval, prob_gnn, "GNN") +
                              metricas_por_familia(dataset, idx_eval, prob_base, "línea de base"))
log("Por familia (prueba + fuera de familia):\n" + tabla_familias.to_string(index=False))
tabla_familias.to_csv(os.path.join(RES, "gnn_por_familia.csv"), index=False)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for modelo, prob, estilo in [("línea de base", prob_base, "--"), ("GNN", prob_gnn, "-")]:
    for parte, idx in [("prueba", idx_prueba), ("fuera de familia", idx_fuera)]:
        if 0 < y[idx].sum() < len(idx):
            fpr, tpr, _ = roc_curve(y[idx], prob[idx])
            ax[0].plot(fpr, tpr, estilo, label=f"{modelo} · {parte} (AUC {filas[(modelo, parte)]['AUC']:.2f})")
ax[0].plot([0, 1], [0, 1], ":", color="0.6")
ax[0].set_xlabel("falsos positivos"); ax[0].set_ylabel("verdaderos positivos"); ax[0].set_title("Curvas ROC")
ax[0].legend(fontsize=8)
tabla_metricas.unstack(0)["AUC"].plot.bar(ax=ax[1], rot=0)
ax[1].axhline(0.5, ls=":", color="0.6"); ax[1].set_ylim(0, 1); ax[1].set_title("AUC: GNN vs línea de base")
plt.tight_layout(); plt.savefig(os.path.join(RES, "gnn_verificacion.png"), dpi=120); plt.show()


def certificar_con_fase(G, fase):
    A = nx.to_numpy_array(G, nodelist=range(G.number_of_nodes()))
    for th in descenso(A, np.vstack([fase, -fase]), ITER_DESCENSO):
        if diferencias_en_aristas(A, th).max() > 1e-3:
            est = analizar_estado(A, th)
            if est["estricto"] and not est["sincronizado"]:
                return True
    return False


positivos = [i for i in np.concatenate([idx_prueba, idx_fuera]) if y[i] == 1]
exitos = sum(certificar_con_fase(dataset[i]["G"], fases_gnn[i]) for i in positivos)
cuenca = np.mean([dataset[i]["cuenca"] for i in positivos]) if positivos else float("nan")
log(f"Positivos de prueba/fuera: {len(positivos)}. Un arranque desde la fase de la GNN certifica "
    f"{exitos}/{len(positivos)} = {exitos / max(1, len(positivos)):.2f}; un arranque aleatorio: {cuenca:.2f}")

# %% [markdown]
# **Reentrenamiento final.** Para usarla en el annealing, la GNN se ajusta un poco más con *todo* el dataset
# (incluidas las familias reservadas, que son justamente puntos de partida del annealing).

# %%
entrenar(gnn, np.arange(len(dataset)), EPOCAS_GNN_FINAL)
torch.save(gnn.state_dict(), os.path.join(RES, "gnn_pipeline.pt"))
log("GNN final → resultados/gnn_pipeline.pt")

# %% [markdown]
# ## 4. Paso 1c: simulated annealing guiado por la GNN
#
# Cada paso agrega o saca una arista (manteniendo el grafo conexo). Puntaje:
# `PESO_MU_SA · μ(G) + BETA_SA · log p_GNN(G)`, con `PESO_MU_SA` bajo porque no exigimos densidad alta.
# Cuando la GNN da p > `UMBRAL_GNN` en un grafo nuevo, se llama al etiquetador, pasándole además la fase
# propuesta por la GNN como semilla.
#
# **Puntos de partida** (n entre `N_MIN_SA` y `N_MAX_SA`): circulantes (twisted), blow-ups de C₄, blow-ups de
# ciclos de 5 a 8, Cayley no circulantes, positivos del dataset y Erdős–Rényi densos.
#
# **Diversidad**: la *firma* de un θ* es (|⟨e^{iℓθ}⟩|, ℓ = 1..6), invariante por rotación y reflexión. Dos
# hallazgos con el mismo n y firmas a distancia < `TOL_DIVERSIDAD` se consideran el mismo patrón.

# %%
def firma(th):
    return np.array([np.abs(np.exp(1j * l * th).mean()) for l in range(1, 7)])


def inicio_sa(familia, rng):
    while True:
        n = int(rng.integers(N_MIN_SA, N_MAX_SA + 1))
        if familia == "circulante":
            G = nx.circulant_graph(n, range(1, int(rng.integers(1, max(2, n // 4))) + 1))
        elif familia == "blowup_C4":
            G = blowup_ciclo(4, rng, tam_max=max(2, n // 4))
        elif familia == "blowup_C5_8":
            c = int(rng.integers(5, 9))
            G = blowup_ciclo(c, rng, tam_max=max(2, n // c))
        elif familia == "cayley":
            G = cayley_aleatorio(rng, (N_MIN_SA, N_MAX_SA))
        elif familia == "er_denso":
            G = nx.gnp_random_graph(n, 0.5, seed=int(rng.integers(1e9)))
        elif familia == "dataset_positivos":
            pos = [r for r in dataset if r["etiqueta"] == "no_sync_cert" and N_MIN_SA <= r["n"] <= N_MAX_SA]
            pos.sort(key=lambda r: -mu_grafo(r["G"]))
            G = pos[int(rng.integers(min(len(pos), 20)))]["G"].copy()
        if nx.is_connected(G) and N_MIN_SA <= G.number_of_nodes() <= N_MAX_SA:
            return nx.convert_node_labels_to_integers(G)


def recocido(G, familia, estado_gnn, semilla):
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    modelo = GNNKuramoto(); modelo.load_state_dict(estado_gnn)
    rng_l = np.random.default_rng(semilla)
    n = G.number_of_nodes()
    p, f = predecir(modelo, [G]); p, fase = p[0], f[0]
    puntaje = lambda G, p: PESO_MU_SA * mu_grafo(G) + BETA_SA * np.log(p + 1e-6)
    actual = puntaje(G, p)
    hallazgos, verificados = [], set()
    for paso in range(PASOS_SA + 1):
        if paso > 0:
            H = G.copy()
            u, v = rng_l.choice(n, 2, replace=False)
            H.remove_edge(u, v) if H.has_edge(u, v) else H.add_edge(u, v)
            if not nx.is_connected(H):
                continue
            pH, fH = predecir(modelo, [H]); pH, fH = pH[0], fH[0]
            nuevo = puntaje(H, pH)
            if not (nuevo > actual or rng_l.random() < np.exp((nuevo - actual) / TEMPERATURA_SA)):
                continue
            G, actual, p, fase = H, nuevo, pH, fH
        clave = nx.weisfeiler_lehman_graph_hash(G)
        if p > UMBRAL_GNN and clave not in verificados:
            verificados.add(clave)
            r = etiquetar(G, rng_l, semillas_extra=np.vstack([fase, -fase]))
            if r["etiqueta"] == "no_sync_cert":
                hallazgos.append(dict(G=G.copy(), familia=familia, n=n, mu_G=mu_grafo(G), p_gnn=p, **r))
    return hallazgos, len(verificados)


FAMILIAS_SA = ["circulante", "blowup_C4", "blowup_C5_8", "cayley", "dataset_positivos", "er_denso"]
trabajos = [(inicio_sa(fam, rng), fam, int(rng.integers(1e9))
            ) for fam in FAMILIAS_SA for _ in range(CORRIDAS_POR_FAMILIA)]
estado_gnn = {k: v.clone() for k, v in gnn.state_dict().items()}
t0 = time.time()
salidas = Parallel(n_jobs=N_JOBS)(delayed(recocido)(G, fam, estado_gnn, s) for G, fam, s in trabajos)
hallazgos = [h for hs, _ in salidas for h in hs]
llamadas = sum(c for _, c in salidas)
log(f"annealing: {len(trabajos)} corridas, {llamadas} llamadas al etiquetador, "
    f"{len(hallazgos)} mínimos espurios certificados ({time.time() - t0:.0f} s)")
print(pd.DataFrame([dict(familia=h["familia"], origen=h["origen"]) for h in hallazgos]).value_counts()
      if hallazgos else "sin hallazgos")

# %% [markdown]
# ## 5. Paso 2: extraer θ* (mínimo espurio certificado)
#
# Cada hallazgo ya trae su θ* (lo da el etiquetador: descenso + Newton + Hessiano). Acá se eliminan
# repetidos (isomorfos o con el mismo patrón de fases), se agregan las **semillas de control** y se
# arma la lista de candidatos.
#
# **Controles** (no son para buscar, son para verificar el pipeline):
# - `control_twisted_C19`: el circulante C₁₉(1..5). El primer SDP debería dar ≈ 0.6839 (menos el efecto de δ).
# - `control_okeeffe_k7`: la configuración de 7 clusters de la geometría de O'Keeffe. La semilla es ese diseño
#   optimizado con δ = 10⁻⁴ (μ = 0.6821, `datos/control_okeeffe_k7_d1e-4.json`), con la diagonal de P en 1. El diseño
#   optimizado con δ = 10⁻² (`datos/control_okeeffe_k7.json`) tiene otra geometría (7 clusters casi equiespaciados, sin
#   mitades) y desde ahí CMA-ES no llega a la de cuatro grupos partidos: se queda en ≈ 0.675.

# %%
hallazgos.sort(key=lambda h: -h["mu_G"])
candidatos, claves = [], set()
for h in hallazgos:
    clave = nx.weisfeiler_lehman_graph_hash(h["G"])
    f = firma(h["theta"])
    if clave in claves or any(c["n"] == h["n"] and np.linalg.norm(c["firma"] - f) < TOL_DIVERSIDAD
                              for c in candidatos):
        continue
    claves.add(clave)
    candidatos.append(dict(h, firma=f))
log(f"hallazgos: {len(hallazgos)} → sin repetir: {len(candidatos)}")
# no se corta por μ(G): los MAX_CANDIDATOS se eligen por μ pulido al principio de la sección 7

G19 = nx.circulant_graph(19, range(1, 6))
r19 = etiquetar(G19, rng)
assert r19["etiqueta"] == "no_sync_cert", "el control twisted debería tener mínimo espurio"
candidatos.append(dict(G=G19, familia="control_twisted_C19", n=19, mu_G=mu_grafo(G19), p_gnn=None,
                       firma=firma(r19["theta"]), **r19))

for i, c in enumerate(candidatos):
    c["id"] = f"c{i:03d}"
    c["A"] = nx.to_numpy_array(c["G"], nodelist=range(c["n"]))

with open(os.path.join(DATOS, "control_okeeffe_k7_d1e-4.json")) as f:
    ok = json.load(f)
control_ok = dict(id="control_okeeffe_k7", familia="control_okeeffe_k7", n=7, G=None, A=None,
                  x0=np.array(ok["x"]), phi0=np.array(ok["phi"]), P0=np.array(ok["P"]), mu_G=None)
np.fill_diagonal(control_ok["P0"], 1.0)                 # diagonal en 1 también en la semilla

tabla_cand = pd.DataFrame([dict(id=c["id"], familia=c["familia"], n=c["n"], mu_G=round(c["mu_G"], 4),
                                lam2=round(c["lam2"], 4), cuenca=round(c["cuenca"], 3), origen=c["origen"])
                           for c in candidatos])
log("\n" + tabla_cand.to_string(index=False))
guardar(candidatos, "candidatos_paso2.pkl")

fig, axs = plt.subplots(1, min(4, len(candidatos)), figsize=(3 * min(4, len(candidatos)), 3), squeeze=False)
for ax, c in zip(axs[0], candidatos[:4]):
    th = c["theta"]
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.7"))
    ax.scatter(np.cos(th), np.sin(th), s=40, zorder=3)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(f"{c['id']} ({c['familia']})\nμ(G)={c['mu_G']:.3f}", fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. Primer SDP: cada nodo es un cluster
#
# **Modelo de clusters.** k grupos; el grupo a tiene tamaño x_a y fase φ_a, y P = (p_ab) son las densidades
# entre grupos (p_aa es la densidad interna). En el límite de clusters grandes:
#
# - equilibrio: $F_a=\sum_b p_{ab}x_b\sin(\varphi_b-\varphi_a)=0$
# - conectividad: $\mu=\min_a\sum_b p_{ab}x_b$
# - estabilidad dentro de cada cluster: $D_a=\sum_b p_{ab}x_b\cos(\varphi_a-\varphi_b)\ge\delta$
# - estabilidad entre clusters: $\lambda_2\bigl(X^{-1/2}L(w)X^{-1/2}\bigr)\ge\delta$, con $w_{ab}=x_ax_bp_{ab}\cos(\varphi_a-\varphi_b)$
#
# Con x y φ fijos todo es lineal en P salvo la última condición, que es una LMI: **es un SDP**.
#
# Con `DIAGONAL_UNO = True` se impone además $p_{aa}=1$: dentro de cada cluster todos están conectados con todos.
# Sólo se optimizan las densidades entre clusters distintos.
#
# **Punto de partida (paso 3.1):** k = n, φ = θ*, x_a = 1/n, P = A + I. La diagonal en 1 representa la
# conexión interna completa de cada cluster y hace que el punto de partida reproduzca exactamente
# μ(G) = (δ_min + 1)/n. Ese punto ya cumple el equilibrio y la estabilidad entre clusters (es el Hessiano de G),
# así que el SDP sólo puede mejorar μ(G) si el margen alcanza para `DELTA_SDP1`.
#
# **El μ del primer SDP no decide quién sigue.** Se calcula para todos (tabla e histograma, con `UMBRAL_SDP1` como
# referencia), los twisted se apartan para patearlos y el resto pasa a la **selección por pulido** de la sección 7.

# %%
import cvxpy as cp


class SDPClusters:
    """max t  s.a.  equilibrio, grado ≥ t, D_a ≥ δ, LMI ≥ δ, 0 ≤ P ≤ 1   (x, φ fijos)."""

    def __init__(self, k, delta, margen=False):
        self.k, self.delta, self.margen = k, delta, margen
        self.P = cp.Variable((k, k), symmetric=True)
        self.t = cp.Variable()
        self.Aeq = cp.Parameter((k - 1, k)); self.Xr = cp.Parameter((k, k), nonneg=True)
        self.Cx = cp.Parameter((k, k)); self.K1 = cp.Parameter((k, k)); self.K2 = cp.Parameter((k, k))
        self.UU = cp.Parameter((k, k), PSD=True)
        P = self.P
        Ms = -cp.multiply(self.K1, P) + cp.diag(cp.sum(cp.multiply(self.K2, P), axis=1))
        Ms = (Ms + Ms.T) / 2
        eq = cp.sum(cp.multiply(self.Aeq, P[1:, :]), axis=1) == 0
        diag = [cp.diag(P) == 1] if DIAGONAL_UNO else []
        if margen:      # versión auxiliar: máximo margen alcanzable (guía al CMA-ES cuando es infactible)
            s = self.t
            cons = [eq, cp.sum(cp.multiply(self.Cx, P), axis=1) >= s, P >= 0, P <= 1,
                    Ms + s * self.UU - s * np.eye(k) >> 0] + diag
        else:
            cons = [eq, cp.sum(cp.multiply(self.Xr, P), axis=1) >= self.t,
                    cp.sum(cp.multiply(self.Cx, P), axis=1) >= delta,
                    Ms + delta * self.UU >> delta * np.eye(k), P >= 0, P <= 1] + diag
        self.prob = cp.Problem(cp.Maximize(self.t), cons)

    def resolver(self, x, phi):
        k = self.k
        d = phi[None, :] - phi[:, None]
        S, C = np.sin(d), np.cos(d)
        self.Aeq.value = (S * x[None, :])[1:]
        self.Xr.value = np.tile(x, (k, 1))
        self.Cx.value = C * x[None, :]
        K1 = np.sqrt(np.outer(x, x)) * C; np.fill_diagonal(K1, 0)
        K2 = C * x[None, :]; np.fill_diagonal(K2, 0)
        self.K1.value, self.K2.value = K1, K2
        self.UU.value = np.outer(np.sqrt(x), np.sqrt(x))
        try:
            self.prob.solve(solver=cp.CLARABEL)
        except Exception:
            return None, None
        if self.prob.status not in ("optimal", "optimal_inaccurate") or self.t.value is None:
            return None, None
        P = np.clip((self.P.value + self.P.value.T) / 2, 0, 1)
        if DIAGONAL_UNO:
            np.fill_diagonal(P, 1.0)
        return float(self.t.value), P


def cantidades(x, phi, P):
    """Chequeo independiente (numpy): equilibrio, conectividad, D, λ₂ reducido, parámetro de orden."""
    d = phi[None, :] - phi[:, None]
    S, C = np.sin(d), np.cos(d)
    F = (P * S) @ x
    grado = P @ x
    D = (P * C) @ x
    W = np.outer(x, x) * P * C
    np.fill_diagonal(W, 0)
    L = np.diag(W.sum(1)) - W
    s = 1 / np.sqrt(x)
    ev = np.linalg.eigvalsh(s[:, None] * L * s[None, :])
    return dict(mu=grado.min(), F=np.abs(F).max(), Dmin=D.min(), lam2=ev[1],
                r=np.abs(np.sum(x * np.exp(1j * phi))))


def factible(q, delta, tol_eq=1e-6):
    return q["F"] < tol_eq and q["Dmin"] >= 0.5 * delta and q["lam2"] >= 0.5 * delta


def primer_sdp(c):
    warnings.filterwarnings("ignore")
    if c.get("A") is not None:
        n = c["n"]
        x0, phi0, P0 = np.full(n, 1 / n), c["theta"] - c["theta"][0], c["A"] + np.eye(n)
    else:
        x0, phi0, P0 = c["x0"], c["phi0"] - c["phi0"][0], c["P0"]
    k = len(x0)
    q0 = cantidades(x0, phi0, P0)
    t0 = time.time()
    mu1, P1 = SDPClusters(k, DELTA_SDP1).resolver(x0, phi0)
    q1 = cantidades(x0, phi0, P1) if P1 is not None else None
    ok = q1 is not None and factible(q1, DELTA_SDP1)
    return dict(id=c["id"], k=k, x0=x0, phi0=phi0, P0=P0, q0=q0, mu_antes=q0["mu"],
                mu1=q1["mu"] if ok else np.nan, P1=P1 if ok else None, q1=q1, factible=ok,
                cambio_P=None if not ok else float(np.abs(P1 - P0).sum() / 2), segundos=time.time() - t0)


t0 = time.time()
todos = candidatos + [control_ok]
res1 = Parallel(n_jobs=N_JOBS)(delayed(primer_sdp)(c) for c in todos)
for c, r in zip(todos, res1):
    c.update(sdp1=r)
log(f"primer SDP: {len(res1)} candidatos en {time.time() - t0:.0f} s")
tabla1 = pd.DataFrame([dict(id=r["id"], k=r["k"], mu_antes=round(r["mu_antes"], 4), mu_SDP1=round(r["mu1"], 4),
                            lam2_inicial=f"{r['q0']['lam2']:.2e}", Dmin_inicial=f"{r['q0']['Dmin']:.2e}",
                            factible=r["factible"], suma_cambios_P=r["cambio_P"]) for r in res1])
log("\n" + tabla1.to_string(index=False))
r19 = next(r for r in res1 if r["id"] == next(c["id"] for c in candidatos if c["familia"] == "control_twisted_C19"))
log(f"CONTROL twisted C19: μ antes {r19['mu_antes']:.4f} → SDP {r19['mu1']:.5f}  (esperado ≈ 0.6839 − O(δ))")

# --- twisted: se apartan antes del filtro 1 ---
def es_twisted_aprox(x, phi, tol=None):
    """Fases equiespaciadas y tamaños iguales, con tolerancia relativa. Antes se juntan las fases que coinciden
    (twisted con mcd(q, n) > 1, donde varios nodos comparten fase)."""
    tol = TOL_TWISTED if tol is None else tol
    a = np.mod(phi, 2 * np.pi); o = np.argsort(a); a, xs = a[o], np.asarray(x)[o]
    grupos_x, actual, ref = [], xs[0], a[0]
    for v, w in zip(a[1:], xs[1:]):
        if v - ref < 1e-3:
            actual += w
        else:
            grupos_x.append(actual); actual, ref = w, v
    if grupos_x and a[0] + 2 * np.pi - ref < 1e-3:
        grupos_x[0] += actual
    else:
        grupos_x.append(actual)
    g = len(grupos_x)
    if g < 3:
        return False
    u = np.unique(np.round(a, 3))
    u = u[np.concatenate([[True], np.diff(u) > 1e-3])]
    if len(u) != g:
        return False
    gaps = np.diff(np.concatenate([u, [u[0] + 2 * np.pi]]))
    return bool(np.abs(gaps * g / (2 * np.pi) - 1).max() < tol and np.abs(np.array(grupos_x) * g - 1).max() < tol)


fuentes_patada = []                       # (id de origen, x, φ) de cada twisted a patear
twisted_paso2 = []
if TRATAR_TWISTED != "nada":
    for c in todos:
        if c["familia"].startswith("control") or c.get("A") is None:
            continue
        if es_twisted_aprox(np.full(c["n"], 1 / c["n"]), c["theta"]):
            twisted_paso2.append(c["id"])
            if TRATAR_TWISTED == "patada":
                fuentes_patada.append(dict(padre=c["id"], x=np.full(c["n"], 1 / c["n"]), phi=c["theta"].copy(),
                                           etapa="θ* del paso 2"))
log(f"TWISTED en el paso 2 ({TRATAR_TWISTED}): {len(twisted_paso2)} de {len(todos)} candidatos tienen θ* twisted "
    f"{twisted_paso2}")

# --- la selección es por pulido (principio de la sección 7), no por el μ del primer SDP ---
siguen = [c for c in todos if c["familia"].startswith("control") and c["sdp1"]["factible"]]
preseleccion = [c for c in todos if not c["familia"].startswith("control") and c["id"] not in twisted_paso2]
log(f"{len(preseleccion)} candidatos no twisted pasan a la selección por pulido; "
    f"controles que siguen: {[c['id'] for c in siguen]}")

def lineas_cotas(a, valores, umbral=None):
    """Cotas de Canale y O'Keeffe, umbral del filtro y el mayor μ del histograma, escrito en el gráfico."""
    valores = [v for v in valores if np.isfinite(v)]
    if umbral is not None:
        a.axvline(umbral, ls="--", color="k", lw=1, label=f"umbral {umbral:g}")
    a.axvline(COTA_CANALE, color="tab:green", lw=1.2, label=f"Canale 11/16 = {COTA_CANALE:.4f}")
    a.axvline(M_OBJETIVO, color="tab:red", lw=1.2, label=f"O'Keeffe {M_OBJETIVO}")
    if valores:
        mx = max(valores)
        a.axvline(mx, color="tab:purple", ls=":", lw=1.2)
        a.text(0.02, 0.95, f"máx μ = {mx:.6f}", transform=a.transAxes, va="top", fontsize=9,
               color="tab:purple", fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.8))
    a.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 0.86))


fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
bins = np.linspace(0, 1, 41)
ax[0].hist([r["mu_antes"] for r in res1], bins=bins); ax[0].set_title("μ antes del primer SDP (μ(G))")
ax[1].hist([r["mu1"] for r in res1 if r["factible"]], bins=bins, color="C1"); ax[1].set_title("μ después del primer SDP")
lineas_cotas(ax[0], [r["mu_antes"] for r in res1], UMBRAL_SDP1)
lineas_cotas(ax[1], [r["mu1"] for r in res1 if r["factible"]], UMBRAL_SDP1)
for a in ax:
    a.set_xlabel("μ"); a.set_ylabel("candidatos")
plt.tight_layout(); plt.savefig(os.path.join(RES, "hist_sdp1.png"), dpi=120); plt.show()

# %% [markdown]
# ## 7. CMA-ES + SDP con el filtro TOPE
#
# **Dos niveles.** Adentro, para (x, φ) dados, el SDP da las mejores densidades. Afuera, CMA-ES mueve
# (x, φ) para maximizar el μ del SDP. Variables: z = (s_1..s_k, φ_2..φ_k), con
# x = X_MIN_FRAC/k + (1 − X_MIN_FRAC)·softmax(s) y φ_1 = 0. Si para algún z el SDP es infactible, el valor es
# una penalidad que crece con lo que falta de margen (un SDP auxiliar calcula el máximo margen alcanzable);
# así CMA-ES puede volver a la región factible. También se penaliza |Σ x e^{iφ}| > `R_MAX`, para que las
# fases no colapsen al estado sincronizado.
#
# **Escalera de márgenes** `DELTAS`. Cada nivel arranca del óptimo del anterior (continuación). Como al bajar δ
# las restricciones se relajan, μ nunca baja de un nivel al siguiente.
#
# **Filtro TOPE** (Techo Optimista y Pareto Escalonado):
# 1. *Nivel 0* (δ = `DELTAS[0]`): se descartan los infactibles y los que quedan más de `MARGEN_NIVEL0` por
#    debajo del mejor.
# 2. *Techo optimista* (desde el nivel 1): techo = μ_k + L·Δ_k, con Δ_k = μ_k − μ_{k−1} y L los niveles que
#    faltan. Se descarta si techo < `M_OBJETIVO`. Supuesto único: las ganancias no crecen al bajar δ.
# 3. *Alarma*: si Δ_k > Δ_{k−1} el supuesto no vale; el candidato no se descarta y queda marcado.
# 4. *Frente de Pareto* en (μ, Δ): los no dominados siguen con el presupuesto multiplicado por
#    `FACTOR_PRESUPUESTO`; los dominados quedan **en espera** con su estado guardado.
# 5. *Rescate*: si al final ninguno supera `M_OBJETIVO`, se retoman los que quedaron en espera.
#
# Los **controles** recorren toda la escalera sin descartes ni Pareto, para poder compararlos con lo esperado.
#
# Dentro de cada nivel, una corrida se corta si en `VENTANA_ESTANCAMIENTO` generaciones el mejor μ no sube
# más de `TOL_MEJORA`. Todos los descartes se guardan en `resultados/descartados_tope.pkl`.
#
# **Selección de los candidatos de grafos.** Antes de TOPE, cada candidato no twisted se pule
# `PASOS_SELECCION_GRAFOS` pasos en δ = `DELTA_INICIO_SEMILLAS` desde (x = 1/n, φ = θ*). Entran a TOPE los
# `MAX_CANDIDATOS` con mayor μ pulido, desde el punto pulido. Los demás se guardan en `apartados_seleccion.pkl`.
# Como las semillas, los candidatos de grafos tienen la tolerancia `MARGEN_TECHO_SEMILLAS` en el techo.

# %%
import cma


def decodificar(z, k):
    s = z[:k]
    e = np.exp(s - s.max())
    x = X_MIN_FRAC / k + (1 - X_MIN_FRAC) * e / e.sum()
    return x, np.concatenate([[0.0], z[k:]])


def codificar(x, phi):
    k = len(x)
    s = np.log(np.maximum((x - X_MIN_FRAC / k) / (1 - X_MIN_FRAC), 1e-12))
    return np.concatenate([s - s.mean(), (phi - phi[0])[1:]])


def evaluar(z, k, sdp, sdp_margen, delta):
    x, phi = decodificar(z, k)
    r = np.abs(np.sum(x * np.exp(1j * phi)))
    if r > R_MAX:
        return -3.0 - (r - R_MAX), None
    mu, P = sdp.resolver(x, phi)
    if mu is not None:
        q = cantidades(x, phi, P)
        if factible(q, delta):
            return q["mu"], P
    s, _ = sdp_margen.resolver(x, phi)
    return (-2.0 if s is None else -1.0 + 10 * min(0.0, s - delta)), None


def correr_nivel(est, delta, gens, sigma, semilla):
    """Un nivel de CMA-ES + SDP para un candidato. est = estado del candidato (z, k)."""
    warnings.filterwarnings("ignore")
    k, z0 = est["k"], est["z"]
    sdp, sdp_m = SDPClusters(k, delta), SDPClusters(k, delta, margen=True)
    t0 = time.time()
    mejor_v, _ = evaluar(z0, k, sdp, sdp_m, delta)
    mejor_z, historia, evals = z0.copy(), [mejor_v], 1
    es = cma.CMAEvolutionStrategy(z0, sigma, {"maxiter": gens, "verbose": -9, "seed": semilla,
                                             "tolx": 1e-13, "tolfun": 1e-14})
    while not es.stop():
        Z = es.ask()
        V = [evaluar(np.asarray(z), k, sdp, sdp_m, delta)[0] for z in Z]
        evals += len(Z)
        es.tell(Z, [-v for v in V])
        i = int(np.argmax(V))
        if V[i] > mejor_v:
            mejor_v, mejor_z = V[i], np.asarray(Z[i]).copy()
        historia.append(mejor_v)
        if len(historia) > VENTANA_ESTANCAMIENTO and historia[-1] - historia[-1 - VENTANA_ESTANCAMIENTO] < TOL_MEJORA:
            break
    mu, P = evaluar(mejor_z, k, sdp, sdp_m, delta)
    x, phi = decodificar(mejor_z, k)
    q = cantidades(x, phi, P) if P is not None else None
    return dict(z=mejor_z, mu=mu if P is not None else np.nan, P=P, x=x, phi=phi, q=q, delta=delta,
                generaciones=len(historia) - 1, evaluaciones=evals, segundos=time.time() - t0, historia=historia)


def correr_nivel_reinicios(est, delta, gens, sigma, semilla):
    """varios CMA-ES seguidos con σ decrecientes (`REINICIOS_SIGMA`), cada uno desde el mejor punto.
    En el control de O'Keeffe, 40 generaciones con corte por estancamiento a las 15 dejaban μ en 0.6821 con
    δ = 10⁻⁴; con 3 × 300 generaciones llega a 0.6869. Para k > `K_GRANDE` se hace una sola corrida corta."""
    if est["k"] > K_GRANDE:
        return correr_nivel(est, delta, max(GENS_K_GRANDE, gens // 5), sigma, semilla)
    mejor, z, total_g, total_e, seg, hist = None, est["z"], 0, 0, 0.0, []
    for j, sg in enumerate(REINICIOS_SIGMA):
        u = correr_nivel(dict(k=est["k"], z=z), delta, gens, sg, semilla + j)
        total_g += u["generaciones"]; total_e += u["evaluaciones"]; seg += u["segundos"]; hist += u["historia"]
        if u["P"] is not None and (mejor is None or u["mu"] > mejor["mu"]):
            mejor, z = u, u["z"]
    mejor = mejor or u
    mejor.update(generaciones=total_g, evaluaciones=total_e, segundos=seg, historia=hist)
    return mejor


def paso_lineal(x0, phi0, P0, delta, r, xmin):
    """un SDP con todo linealizado alrededor de (x0, φ0, P0) y región de confianza r.
    Variables: Δx = d, Δφ = e, ΔP = Q y t. Devuelve (x, φ) propuestos o None."""
    k = len(x0)
    dd = phi0[None, :] - phi0[:, None]                         # φ_b − φ_a
    S0, C0 = np.sin(dd), np.cos(dd)
    d = cp.Variable(k); e = cp.Variable(k); Q = cp.Variable((k, k), symmetric=True); t = cp.Variable()
    uno = np.ones((k, 1))
    E = uno @ cp.reshape(e, (1, k)) - cp.reshape(e, (k, 1)) @ uno.T    # e_b − e_a
    X0 = np.tile(x0, (k, 1))                                   # x0_b en la columna b
    Dm = uno @ cp.reshape(d, (1, k))                           # d_b en la columna b
    grado = (P0 * X0).sum(1) + cp.sum(cp.multiply(Q, X0) + cp.multiply(P0, Dm), axis=1)
    F = (P0 * X0 * S0).sum(1) + cp.sum(cp.multiply(Q, X0 * S0) + cp.multiply(P0 * S0, Dm)
                                        + cp.multiply(P0 * X0 * C0, E), axis=1)
    D = (P0 * X0 * C0).sum(1) + cp.sum(cp.multiply(Q, X0 * C0) + cp.multiply(P0 * C0, Dm)
                                        - cp.multiply(P0 * X0 * S0, E), axis=1)
    # M = X^-1/2 L X^-1/2: fuera de la diagonal −√(x_a x_b) p_ab cos, en la diagonal Σ_{b≠a} x_b p_ab cos
    rr = np.sqrt(np.outer(x0, x0)); off = 1 - np.eye(k)
    dr = cp.multiply(d, 1 / x0)
    Dr = 0.5 * (cp.reshape(dr, (k, 1)) @ uno.T + uno @ cp.reshape(dr, (1, k)))
    Moff = -(rr * P0 * C0 * off + cp.multiply(Q, rr * C0 * off) + cp.multiply(rr * P0 * C0 * off, Dr)
             - cp.multiply(rr * P0 * S0 * off, E))
    diag = (P0 * X0 * C0 * off).sum(1) + cp.sum(cp.multiply(Q, X0 * C0 * off) + cp.multiply(P0 * C0 * off, Dm)
                                                 - cp.multiply(P0 * X0 * S0 * off, E), axis=1)
    M = Moff + cp.diag(diag)
    s0 = np.sqrt(x0); ds = cp.multiply(d, 1 / (2 * s0))
    UU = np.outer(s0, s0) + cp.reshape(ds, (k, 1)) @ s0[None, :] + s0[:, None] @ cp.reshape(ds, (1, k))
    Ms, UUs = (M + M.T) / 2, (UU + UU.T) / 2
    cons = [grado >= t, F[1:] == 0, D >= delta, Ms + delta * UUs - delta * np.eye(k) >> 0,
            cp.sum(d) == 0, x0 + d >= xmin, e[0] == 0, P0 + Q >= 0, P0 + Q <= 1,
            cp.abs(e) <= r, cp.abs(d) <= r * x0, cp.abs(Q) <= min(1.0, 10 * r)]
    if DIAGONAL_UNO:
        cons.append(cp.diag(Q) == 0)
    pr = cp.Problem(cp.Maximize(t), cons)
    try:
        pr.solve(solver=cp.CLARABEL)
    except Exception:
        return None
    if pr.status not in ("optimal", "optimal_inaccurate") or d.value is None:
        return None
    return x0 + d.value, phi0 + e.value


def pulir(x, phi, delta, iters=None, r0=None):
    """pulido convexo secuencial desde (x, φ). En cada paso se re-linealiza en el punto actual; la
    propuesta se acepta sólo si el SDP exacto en el nuevo (x, φ) es factible y sube μ (Δ = 0 siempre es
    factible, así que μ nunca baja). Si se acepta, la región de confianza se duplica; si no, se divide por 3.
    Termina cuando la región es menor que `RADIO_MIN_PULIDO` o μ no sube en 30 pasos."""
    warnings.filterwarnings("ignore")
    iters = ITERS_PULIDO if iters is None else iters
    k = len(x); xmin = X_MIN_FRAC / k
    sdp = SDPClusters(k, delta)
    t0 = time.time()
    _, P = sdp.resolver(x, phi)
    if P is None or not factible(cantidades(x, phi, P), delta):
        return None
    mu = cantidades(x, phi, P)["mu"]
    r, hist, aceptados, it = (RADIO_PULIDO if r0 is None else r0), [mu], 0, -1
    for it in range(iters):
        prop = paso_lineal(x, phi, P, delta, r, xmin)
        ok = False
        if prop is not None:
            xn, phin = prop
            xn = np.maximum(xn, xmin); xn = xn / xn.sum()
            _, Pn = sdp.resolver(xn, phin)
            if Pn is not None:
                q = cantidades(xn, phin, Pn)
                if factible(q, delta) and q["r"] <= R_MAX and q["mu"] > mu + 1e-12:
                    ok = True
                    x, phi, P, mu = xn, phin - phin[0], Pn, q["mu"]
        if ok:
            aceptados += 1; r = min(2 * r, 0.2)
        else:
            r /= 3
        hist.append(mu)
        if r < RADIO_MIN_PULIDO or (len(hist) > 30 and hist[-1] - hist[-31] < 1e-9):
            break
    return dict(x=x, phi=phi, P=P, mu=mu, historia=hist, aceptados=aceptados, iteraciones=it + 1,
                segundos=time.time() - t0)


def correr_nivel_completo(est, delta, gens, sigma, semilla):
    """CMA-ES (elige la cuenca) y después, con `PULIDO_SCP`, el pulido convexo secuencial."""
    u = correr_nivel_reinicios(est, delta, gens, sigma, semilla)
    u["mu_cma"] = u["mu"]
    if not PULIDO_SCP or u["P"] is None:
        return u
    p_ = pulir(u["x"], u["phi"], delta)
    if p_ is not None and p_["mu"] > u["mu"]:
        u.update(x=p_["x"], phi=p_["phi"], P=p_["P"], mu=p_["mu"], q=cantidades(p_["x"], p_["phi"], p_["P"]),
                 z=codificar(p_["x"], p_["phi"]))
    if p_ is not None:
        u.update(pulido_iteraciones=p_["iteraciones"], pulido_aceptados=p_["aceptados"],
                 pulido_segundos=p_["segundos"], historia_pulido=p_["historia"])
        u["segundos"] += p_["segundos"]
    return u


def no_dominados(ids, mu, gan):
    return [i for i in ids if not any(mu[j] >= mu[i] and gan[j] >= gan[i] and (mu[j] > mu[i] or gan[j] > gan[i])
                                      for j in ids if j != i)]


# --- los candidatos de grafos se eligen por μ después de un pulido corto ---
def pulido_seleccion(c):
    warnings.filterwarnings("ignore")
    n = c["n"]
    u = pulir(np.full(n, 1 / n), c["theta"] - c["theta"][0], DELTA_INICIO_SEMILLAS, PASOS_SELECCION_GRAFOS)
    return None if u is None else dict(x=u["x"], phi=u["phi"], P=u["P"], mu=u["mu"])


t0 = time.time()
pul_sel = Parallel(n_jobs=N_JOBS)(delayed(pulido_seleccion)(c) for c in preseleccion)
for c, u in zip(preseleccion, pul_sel):
    c["pulido_sel"] = u
    c["mu_pulido"] = np.nan if u is None else u["mu"]
validos_sel = sorted([c for c in preseleccion if np.isfinite(c["mu_pulido"])], key=lambda c: -c["mu_pulido"])
elegidos = validos_sel[:MAX_CANDIDATOS]
apartados = [c for c in preseleccion if c not in elegidos]
guardar([dict(id=c["id"], familia=c["familia"], G=c.get("G"), theta=c.get("theta"), mu_G=c["mu_G"],
              mu_pulido=c["mu_pulido"], pulido_sel=c["pulido_sel"]) for c in apartados], "apartados_seleccion.pkl")
log(f"SELECCIÓN POR PULIDO ({PASOS_SELECCION_GRAFOS} pasos, δ={DELTA_INICIO_SEMILLAS:g}, {time.time() - t0:.0f} s): "
    f"{len(validos_sel)} de {len(preseleccion)} factibles; entran {len(elegidos)} "
    f"(μ pulido {min([c['mu_pulido'] for c in elegidos], default=np.nan):.5f}–"
    f"{max([c['mu_pulido'] for c in elegidos], default=np.nan):.5f})")
if validos_sel:
    tabla_sel = pd.DataFrame([dict(id=c["id"], familia=c["familia"], n=c["n"], mu_G=round(c["mu_G"], 4),
                                   mu_SDP1=round(c["sdp1"]["mu1"], 4), mu_pulido=round(c["mu_pulido"], 5),
                                   entra=c in elegidos) for c in validos_sel])
    log("\n" + tabla_sel.head(2 * MAX_CANDIDATOS).to_string(index=False))
    tabla_sel.to_csv(os.path.join(RES, "seleccion_grafos.csv"), index=False)

estados = {}
for c in siguen:                                  # controles
    r = c["sdp1"]
    estados[c["id"]] = dict(id=c["id"], familia=c["familia"], k=r["k"], z=codificar(r["x0"], r["phi0"]),
                            mu_sdp1=r["mu1"], niveles=[], presupuesto=GENS_BASE, estado="activo", alarma=False,
                            motivo=None, nivel_inicio=0)
for c in elegidos:                                # candidatos de grafos, desde el punto pulido
    u = c["pulido_sel"]
    estados[c["id"]] = dict(id=c["id"], familia=c["familia"], k=len(u["x"]), z=codificar(u["x"], u["phi"]),
                            mu_sdp1=u["mu"], niveles=[], presupuesto=GENS_BASE, estado="activo", alarma=False,
                            motivo=None, nivel_inicio=0, semilla=True)

descartados, en_espera = [], []
L_TOTAL = len(DELTAS) - 1


def ejecutar_niveles(ids, desde, solo_techo=False, patear=True):
    """Corre la escalera desde el nivel `desde` para los candidatos `ids`, aplicando TOPE.
    Si un candidato converge a twisted, se descarta y (con `patear`) se guarda como fuente de patadas."""
    activos = list(ids)
    for nivel in range(desde, len(DELTAS)):
        if not activos:
            break
        if tiempo_agotado():
            log(f"TIEMPO: pasaron más de {TIEMPO_MAX_HORAS} h; no se corre el nivel {nivel} para {activos}")
            break
        delta = DELTAS[nivel]
        sigma = max(SIGMA_MIN, SIGMA0 * FACTOR_SIGMA ** nivel)
        t0 = time.time()
        salidas = Parallel(n_jobs=N_JOBS)(delayed(correr_nivel_completo)(estados[i], delta, estados[i]["presupuesto"], sigma,
                                                                int(rng.integers(1e9))) for i in activos)
        for i, s in zip(activos, salidas):
            estados[i]["niveles"].append(s)
            if s["P"] is not None:
                estados[i]["z"] = s["z"]
        mus = {i: [n_["mu"] for n_ in estados[i]["niveles"]] for i in activos}
        mu = {i: mus[i][-1] for i in activos}
        # ganancia respecto del nivel anterior (sólo existe desde el nivel 1)
        gan = {i: (mus[i][-1] - mus[i][-2]) if len(mus[i]) > 1 and np.isfinite(mu[i]) else np.nan for i in activos}
        finitos = [v for v in mu.values() if np.isfinite(v)]
        filas, siguen_n = [], []
        for i in activos:
            e = estados[i]
            fila = dict(id=i, mu=round(mu[i], 6), mu_cma=round(e["niveles"][-1].get("mu_cma", np.nan), 6),
                        ganancia=round(gan[i], 6), gens=e["niveles"][-1]["generaciones"],
                        pasos_pulido=e["niveles"][-1].get("pulido_aceptados"), seg=round(e["niveles"][-1]["segundos"], 1))
            ult = e["niveles"][-1]
            if e["familia"].startswith("control"):
                pass                              # los controles nunca se descartan ni entran al frente
            elif TRATAR_TWISTED != "nada" and np.isfinite(mu[i]) and es_twisted_aprox(ult["x"], ult["phi"]):
                e["estado"], e["motivo"] = "descartado", f"convergió a twisted (δ={delta:g})"
                if TRATAR_TWISTED == "patada" and patear:
                    fuentes_patada.append(dict(padre=i, x=ult["x"].copy(), phi=ult["phi"].copy(),
                                               etapa=f"twisted en δ={delta:g}"))
                    e["motivo"] += " → pateado"
            elif not np.isfinite(mu[i]):
                e["estado"], e["motivo"] = "descartado", f"infactible con δ={delta:g}"
            elif nivel == 0 and not solo_techo and mu[i] < max(finitos) - MARGEN_NIVEL0:
                e["estado"], e["motivo"] = "descartado", f"nivel 0: más de {MARGEN_NIVEL0} por debajo del mejor"
            elif np.isfinite(gan[i]):
                L = L_TOTAL - nivel
                techo = mu[i] + L * gan[i]
                fila["techo"] = round(techo, 6)
                if len(mus[i]) > 2 and gan[i] > (mus[i][-2] - mus[i][-3]) + 1e-12:
                    e["alarma"] = True
                    fila["ALARMA"] = "ganancia creciente: revisar a mano"
                margen_t = MARGEN_TECHO_SEMILLAS if (e.get("semilla") or e["familia"] in
                                                     ("sintetica", "patada", "evolucion", "gnn_pesada")) else 0.0
                if L > 0 and techo < M_OBJETIVO - margen_t and not e["alarma"]:
                    e["estado"], e["motivo"] = "descartado", f"techo {techo:.5f} < meta (δ={delta:g})"
            fila["estado"] = e["estado"]
            if e["estado"] == "activo":
                siguen_n.append(i)
            filas.append(fila)
        # frente de Pareto (desde el nivel 1, si no es el último)
        if nivel > 0 and nivel < L_TOTAL and not solo_techo and siguen_n:
            controles = [i for i in siguen_n if estados[i]["familia"].startswith("control")]
            frente = no_dominados([i for i in siguen_n if i not in controles], mu, gan) + controles
            for i in siguen_n:
                if i in frente:
                    estados[i]["presupuesto"] = int(estados[i]["presupuesto"] * FACTOR_PRESUPUESTO)
                else:
                    estados[i]["estado"] = "espera"
                    en_espera.append(i)
            for f_ in filas:
                if f_["id"] in siguen_n:
                    f_["estado"] = estados[f_["id"]]["estado"]
            siguen_n = frente
        for i in activos:
            if estados[i]["estado"] == "descartado" and i not in [d["id"] for d in descartados]:
                descartados.append(estados[i])
        log(f"\n--- nivel {nivel} (δ = {delta:g}), {time.time() - t0:.0f} s ---")
        log(pd.DataFrame(filas).to_string(index=False))
        log(f"siguen {len(siguen_n)}, descartados {sum(estados[i]['estado'] == 'descartado' for i in activos)}, "
            f"en espera {sum(estados[i]['estado'] == 'espera' for i in activos)}")
        guardar(estados, "estado_tope.pkl")
        guardar(descartados, "descartados_tope.pkl")
        activos = siguen_n
    for i in activos:
        estados[i]["estado"] = "final"
    return activos


finales = ejecutar_niveles(list(estados), 0)

# %% [markdown]
# ## 6b. Semillas sintéticas y patadas a los twisted
#
# **Receta de una semilla** (`construir_semilla`): se parte de g grupos de fases con centros y masas.
# 1. Se eligen al azar algunos grupos (de 0 a g) y se parten en 2 partes (o en 3, con probabilidad `PROB_TERCIO`)
#    separadas ±ε, con ε uniforme en `EPS_PARTICION`. La masa del grupo se reparte entre las partes con poco ruido.
# 2. Se exige x_a ≥ `X_MIN_FRAC`/k, se fija φ₁ = 0 y se descarta la semilla si |Σ x e^{iφ}| > `R_MAX`
#    o si tiene más de `K_MAX_SEMILLA` clusters.
#
# **Sintéticas:** centros en 2πj/g más un corrimiento uniforme de hasta ±`JITTER_CENTROS`, con g según
# `PROB_GRUPOS`; masas ~ Dirichlet(α), con α log-uniforme en `ALFA_DIRICHLET_RANGO`. Una fracción
# `FRACCION_OKEEFFE` son "tipo O'Keeffe": 4 grupos, todos partidos en dos mitades.
#
# **Patadas:** a un twisted (x, φ) se le ordenan las fases y se cortan en g arcos contiguos de vecinos (con un punto
# de corte al azar). Cada arco da un grupo con su masa total (con ±15% de ruido) y su fase media. Después sigue la
# misma receta. Así se reutiliza la geometría del twisted en vez de tirarla.
#
# **Selección:** cada semilla se evalúa con el SDP en δ = `DELTA_INICIO_SEMILLAS` (si es infactible, queda afuera).
# El μ del SDP crudo predice mal a dónde llega CMA-ES: la geometría de O'Keeffe hecha a mano da 0.623 crudo y 0.687
# después de CMA-ES. Por eso las `FACTOR_PRESELECCION` × TOP mejores pasan por un **pre-pulido** corto
# (`PASOS_PRESELECCION` pasos del pulido convexo en el mismo δ) y se eligen por el μ pre-pulido. Entran a TOPE las
# mejores de cada familia y las `TOP_PATADAS` patadas con mayor μ, directamente en el nivel de
# `DELTA_INICIO_SEMILLAS`. Si alguna converge a twisted, se descarta sin patearla de nuevo.
#
# **Diez geometrías** (`FAMILIAS_SEMILLAS`, `N_POR_FAMILIA` cada una; de cada una entran las `TOP_POR_FAMILIA` mejores):
#
# | Familia | Receta |
# |---|---|
# | `receta_general` | la receta general de arriba (g = 3..7, 20% tipo O'Keeffe) |
# | `okeeffe` | 4 grupos casi equiespaciados, 3 o 4 de ellos partidos en mitades con ε 0.02–0.15 rad (el control tiene 3 partidos) |
# | `cuatro_dos_partidos` | 4 grupos; dos opuestos partidos en partes desparejas con ε 0.03–0.25 |
# | `tres_grupos` | g = 3, todos partidos en 2 o 3 partes muy abiertas (ε 0.25–0.6 rad) |
# | `cinco_grupos` | g = 5, con grupos partidos en 2 o 3 |
# | `seis_siete` | g = 6 o 7, pocos grupos partidos |
# | `centros_irregulares` | g = 3..6 con centros corridos hasta ±0.6 rad de 2πj/g |
# | `masas_desparejas` | g = 3..6 con masas Dirichlet(α), α en [0.3, 2], y reparto desparejo entre las partes |
# | `tercios` | g = 3..4, la mayoría de los grupos partidos en tres |
# | `dos_escalas` | g = 3..4; cada grupo se parte en dos con ε grande (0.08–0.3) y cada mitad otra vez con ε chico (0.003–0.03) |
#
# Para que el SDP sea factible, los grupos vecinos tienen que quedar a menos de 90° (si no, ningún acoplamiento entre
# ellos tiene cos > 0 y la parte estable queda desconectada): por eso 3 grupos a 120° sólo funcionan muy abiertos, y 4
# grupos a 90° necesitan mitades separadas. Todas pasan por las mismas reglas (x_a ≥ 1.05·`X_MIN_FRAC`/k, |Σ x e^{iφ}| ≤ `R_MAX`, 3 ≤ k ≤ `K_MAX_SEMILLA`).

# %%
def construir_semilla(centros, masas, rng_s, partir_todos=False):
    g = len(centros)
    partir = np.ones(g, bool) if partir_todos else rng_s.random(g) < rng_s.random()   # fracción de grupos partidos
    xs, ps = [], []
    for c_, m_, sp in zip(centros, masas, partir):
        if not sp:
            xs.append(m_); ps.append(c_); continue
        partes = 2 if partir_todos else (3 if rng_s.random() < PROB_TERCIO else 2)
        eps = rng_s.uniform(*EPS_PARTICION)
        offs = np.linspace(-eps, eps, partes)
        w = rng_s.dirichlet(np.full(partes, 20.0)) * m_
        xs += list(w); ps += list(c_ + offs)
    x = np.array(xs); phi = np.array(ps)
    k = len(x)
    if k < 3 or k > K_MAX_SEMILLA:
        return None
    x = x / x.sum()
    xmin = 1.05 * X_MIN_FRAC / k
    for _ in range(20):
        if x.min() >= xmin:
            break
        x = np.maximum(x, xmin); x = x / x.sum()
    phi = np.mod(phi - phi[0], 2 * np.pi)
    if np.abs(np.sum(x * np.exp(1j * phi))) > R_MAX:
        return None
    return x, phi


def semilla_sintetica(rng_s):
    alfa = float(np.exp(rng_s.uniform(*np.log(ALFA_DIRICHLET_RANGO))))
    if rng_s.random() < FRACCION_OKEEFFE:            # tipo O'Keeffe, 4 grupos partidos en mitades
        g = 4
        centros = 2 * np.pi * np.arange(g) / g + rng_s.uniform(-JITTER_CENTROS, JITTER_CENTROS, g)
        return construir_semilla(centros, rng_s.dirichlet(np.full(g, alfa)), rng_s, partir_todos=True)
    g = int(rng_s.choice(GRUPOS_POSIBLES, p=PROB_GRUPOS))
    centros = 2 * np.pi * np.arange(g) / g + rng_s.uniform(-JITTER_CENTROS, JITTER_CENTROS, g)
    return construir_semilla(centros, rng_s.dirichlet(np.full(g, alfa)), rng_s)


def patada(x, phi, rng_s):
    a = np.mod(phi, 2 * np.pi); o = np.argsort(a); a, xs = a[o], np.asarray(x)[o]
    k = len(a)
    opciones = [g for g in GRUPOS_POSIBLES if g < k]
    if not opciones:
        return None
    g = int(rng_s.choice(opciones))
    inicio = int(rng_s.integers(k))
    idx = np.roll(np.arange(k), -inicio)
    cortes = np.sort(rng_s.choice(np.arange(1, k), g - 1, replace=False))
    centros, masas = [], []
    for bloque in np.split(idx, cortes):
        centros.append(np.angle(np.sum(xs[bloque] * np.exp(1j * a[bloque]))))
        masas.append(xs[bloque].sum() * rng_s.uniform(0.85, 1.15))
    return construir_semilla(np.array(centros), np.array(masas), rng_s)


def evaluar_semillas(lote, delta):
    """SDP de cada semilla (x, φ) en δ. Reutiliza el problema cvxpy para cada k."""
    warnings.filterwarnings("ignore")
    cache, salida = {}, []
    for x, phi in lote:
        k = len(x)
        if k not in cache:
            cache[k] = SDPClusters(k, delta)
        mu, P = cache[k].resolver(x, phi)
        ok = mu is not None and factible(cantidades(x, phi, P), delta)
        salida.append((cantidades(x, phi, P)["mu"], P) if ok else (np.nan, None))
    return salida


def seleccionar(semillas, top, delta):
    if not semillas:
        return [], []
    trozos = np.array_split(np.arange(len(semillas)), max(1, min(len(semillas), 4 * max(1, os.cpu_count() or 1))))
    res = Parallel(n_jobs=N_JOBS)(delayed(evaluar_semillas)([(semillas[j]["x"], semillas[j]["phi"]) for j in t], delta)
                                  for t in trozos if len(t))
    res = [r for bloque in res for r in bloque]
    for s_, (mu, P) in zip(semillas, res):
        s_["mu0"], s_["P0"] = mu, P
    validas = sorted([s_ for s_ in semillas if np.isfinite(s_["mu0"])], key=lambda s_: -s_["mu0"])
    crudos = [s_["mu0"] for s_ in validas]
    pre = validas[:FACTOR_PRESELECCION * top]
    # pre-pulido con el pulido convexo (PASOS_PRESELECCION pasos)
    pul = Parallel(n_jobs=N_JOBS)(delayed(pulir)(s_["x"], s_["phi"], delta, PASOS_PRESELECCION) for s_ in pre)
    for s_, u in zip(pre, pul):
        s_["mu_crudo"] = s_["mu0"]
        if u is not None and u["mu"] > s_["mu0"]:
            s_.update(x=u["x"], phi=u["phi"], P0=u["P"], mu0=u["mu"])
    pre.sort(key=lambda s_: -s_["mu0"])
    return pre[:top], crudos


def cerrar_semilla(x, phi):
    """Normaliza un (x, φ) con las mismas reglas que la receta; None si no cumple."""
    k = len(x)
    if k < 3 or k > K_MAX_SEMILLA:
        return None
    x = np.asarray(x, float); x = x / x.sum()
    xmin = 1.05 * X_MIN_FRAC / k
    for _ in range(20):
        if x.min() >= xmin:
            break
        x = np.maximum(x, xmin); x = x / x.sum()
    phi = np.mod(np.asarray(phi, float) - phi[0], 2 * np.pi)
    if np.abs(np.sum(x * np.exp(1j * phi))) > R_MAX:
        return None
    return x, phi


# --- diez familias de geometrías ---
def armar_semilla(centros, masas, partes, eps, rng_s, reparto=20.0):
    """Cada grupo j se parte en partes[j] (1 = no se parte) separadas ±eps[j]; la masa se reparte con Dirichlet(reparto)."""
    xs, ps = [], []
    for c_, m_, pa, ep in zip(centros, masas, partes, eps):
        if pa == 1:
            xs.append(m_); ps.append(c_); continue
        xs += list(rng_s.dirichlet(np.full(pa, reparto)) * m_); ps += list(c_ + np.linspace(-ep, ep, pa))
    return cerrar_semilla(np.array(xs), np.array(ps))


def centros_eq(g, rng_s, jitter=None):
    jitter = JITTER_CENTROS if jitter is None else jitter
    return 2 * np.pi * np.arange(g) / g + rng_s.uniform(-jitter, jitter, g)


def masas_dir(g, rng_s, rango=None):
    alfa = float(np.exp(rng_s.uniform(*np.log(ALFA_DIRICHLET_RANGO if rango is None else rango))))
    return rng_s.dirichlet(np.full(g, alfa))


def partes_al_azar(g, rng_s, prob, tercio=PROB_TERCIO):
    return [(3 if rng_s.random() < tercio else 2) if rng_s.random() < prob else 1 for _ in range(g)]


def semilla_familia(fam, rng_s):
    eps = lambda g, rango=EPS_PARTICION: [rng_s.uniform(*rango) for _ in range(g)]
    if fam == "receta_general":
        return semilla_sintetica(rng_s)
    if fam == "okeeffe":                           # 4 grupos; 3 o 4 partidos en mitades (el control tiene 3 partidos)
        partes = [2] * 4
        if rng_s.random() < 0.5:
            partes[int(rng_s.integers(4))] = 1
        return armar_semilla(centros_eq(4, rng_s), masas_dir(4, rng_s, (5.0, 30.0)), partes, eps(4, (0.02, 0.15)), rng_s)
    if fam == "cuatro_dos_partidos":
        j = int(rng_s.integers(2)); partes = [1] * 4; partes[j] = partes[j + 2] = 2
        return armar_semilla(centros_eq(4, rng_s), masas_dir(4, rng_s, (5.0, 30.0)), partes, eps(4, (0.03, 0.25)), rng_s,
                             reparto=rng_s.uniform(1.0, 5.0))
    if fam == "tres_grupos":                       # con 3 grupos a 120° hace falta abrirlos mucho para que haya vecinos a < 90°
        return armar_semilla(centros_eq(3, rng_s), masas_dir(3, rng_s), [int(rng_s.choice([2, 3])) for _ in range(3)],
                             eps(3, (0.25, 0.6)), rng_s)
    if fam in ("cinco_grupos", "seis_siete"):
        g = 5 if fam == "cinco_grupos" else int(rng_s.choice([6, 7]))
        return armar_semilla(centros_eq(g, rng_s), masas_dir(g, rng_s), partes_al_azar(g, rng_s, 0.6 if g <= 5 else 0.3),
                             eps(g), rng_s)
    if fam == "centros_irregulares":
        g = int(rng_s.integers(3, 7))
        return armar_semilla(centros_eq(g, rng_s, 0.6), masas_dir(g, rng_s), partes_al_azar(g, rng_s, 0.5), eps(g), rng_s)
    if fam == "masas_desparejas":
        g = int(rng_s.integers(3, 7))
        return armar_semilla(centros_eq(g, rng_s), masas_dir(g, rng_s, (0.3, 2.0)), partes_al_azar(g, rng_s, 0.5), eps(g),
                             rng_s, reparto=rng_s.uniform(1.0, 20.0))
    if fam == "tercios":
        g = int(rng_s.integers(3, 5))
        return armar_semilla(centros_eq(g, rng_s), masas_dir(g, rng_s), partes_al_azar(g, rng_s, 0.7, tercio=1.0), eps(g),
                             rng_s)
    if fam == "dos_escalas":
        g = int(rng_s.integers(3, 5)); cs, ms, pa, ep = [], [], [], []
        for c_, m_ in zip(centros_eq(g, rng_s), masas_dir(g, rng_s)):
            if rng_s.random() < 0.6:
                E, f = rng_s.uniform(0.08, 0.3), rng_s.uniform(0.3, 0.7)
                for cc, mm in ((c_ - E, m_ * f), (c_ + E, m_ * (1 - f))):
                    cs.append(cc); ms.append(mm); pa.append(2 if rng_s.random() < 0.6 else 1); ep.append(rng_s.uniform(0.003, 0.03))
            else:
                cs.append(c_); ms.append(m_); pa.append(1); ep.append(0.0)
        return armar_semilla(np.array(cs), np.array(ms), pa, ep, rng_s)
    raise ValueError(fam)


_rng_prueba = np.random.default_rng(SEMILLA + 99)
for fam in FAMILIAS_SEMILLAS:
    ks = [len(r[0]) for r in (semilla_familia(fam, _rng_prueba) for _ in range(200)) if r is not None]
    log(f"  receta {fam:20s}: {len(ks)}/200 válidas, k entre {min(ks, default=0)} y {max(ks, default=0)}")


# %% [markdown]
# ## 6c. GNN entrenada con sistemas de clusters ("grafos pesados")
#
# Un sistema de clusters (x, P) es un grafo pesado de k nodos: el nodo a es un cluster con tamaño x_a y la arista
# ab tiene densidad p_ab. A diferencia de un grafo pesado común, cada cluster tiene además su estabilidad interna
# D_a ≥ δ, así que la etiqueta se define en el modelo de clusters:
#
# **Etiqueta.** El sistema es *positivo* si existen fases φ con F = 0, D_a ≥ `DELTA_ETIQUETA`,
# λ₂(X^{-1/2}L(w)X^{-1/2}) ≥ `DELTA_ETIQUETA` y |Σ x e^{iφ}| ≤ `R_MAX` (un mínimo espurio que el pipeline puede usar).
# El **etiquetador** hace descenso en φ con la dinámica de clusters φ̇_a = Σ_b p_ab x_b sin(φ_b − φ_a) desde
# arranques aleatorios, agrupados (clusters asignados a 3–6 grupos equiespaciados) y twisted, refina con Newton y
# chequea las condiciones con `cantidades`. Devuelve el mínimo con mayor margen min(D_min, λ₂).
#
# **Dataset** (`N_SISTEMAS_PESADOS`, repartido entre cuatro familias y equilibrado 50/50 dentro de cada una):
# - `sdp_sintetica`: semilla sintética (receta de 6a) con las densidades del SDP en un margen log-uniforme en
#   `DELTAS_SDP_DATASET` (por encima de `DELTA_ETIQUETA`, para no quedar en el borde). Casi siempre positiva.
# - `perturbada`: una `sdp_sintetica` con ruido en P (σ log-uniforme entre 0.003 y 0.1) y en x. Da positivos y
#   negativos cerca de la frontera.
# - `aleatoria`: x ~ Dirichlet y P al azar (con densidad media al azar). Casi siempre negativa.
# - `anillo`: p_ab decrece con la distancia circular entre a y b (análogo de los blow-ups de ciclos). Mezcla.
#
# **Red.** La misma arquitectura que la GNN de grafos, con entrada pesada: la agregación usa
# A_ab = p_ab x_b / Σ_c p_ac x_c y los rasgos de cada nodo son k·x_a, el grado pesado Σ_b p_ab x_b, ese grado sobre el
# máximo, los vectores propios del laplaciano normalizado de w = x_a x_b p_ab y rasgos al azar. Parte de los pesos
# de la GNN de grafos (salvo la capa de entrada) y se entrena con pesos por (familia, clase).
#
# **Annealing continuo.** Se mueve en (x, P): cambia una densidad p_ab (ruido de `PASO_P_SA`), los tamaños (ruido
# relativo de `PASO_X_SA`) o parte un cluster en dos. Puntaje: `PESO_MU_SA_PESADO` · μ + `BETA_SA` · log p_GNN, con
# μ = min_a Σ_b p_ab x_b. Cuando la red da p > `UMBRAL_GNN` se llama al etiquetador con la fase de la red como
# semilla extra; los positivos son los **sistemas propuestos**, que entran a TOPE en 6c.

# %%
def ajustar_tamanos(x):
    """Normaliza x y lo lleva a x_a ≥ 1.05·X_MIN_FRAC/k."""
    x = np.asarray(x, dtype=float); x = x / x.sum()
    xmin = 1.05 * X_MIN_FRAC / len(x)
    for _ in range(20):
        if x.min() >= xmin:
            break
        x = np.maximum(x, xmin); x = x / x.sum()
    return x


def fuerza_pesada(x, P, PHI):
    """F_a = Σ_b p_ab x_b sin(φ_b − φ_a) para cada fila de PHI (arranques × k)."""
    W = P * x[None, :]
    c, s = np.cos(PHI), np.sin(PHI)
    return c * (s @ W.T) - s * (c @ W.T)


def jacobiano_pesado(x, P, phi):
    W = P * x[None, :]
    C = W * np.cos(phi[None, :] - phi[:, None])
    np.fill_diagonal(C, 0)
    return C - np.diag(C.sum(1))


def etiquetar_pesado(x, P, rng_l, semillas_extra=None, arranques=None):
    arranques = arranques or ARRANQUES_PESADO
    k = len(x)
    agrupadas = []
    for _ in range(max(1, arranques // 2)):
        g = int(rng_l.choice(GRUPOS_POSIBLES))
        agrupadas.append(2 * np.pi * rng_l.integers(g, size=k) / g + rng_l.normal(0, 0.05, k))
    bloques = [("aleatoria", rng_l.uniform(0, 2 * np.pi, (arranques, k))), ("agrupada", np.array(agrupadas)),
               ("twisted", np.array([2 * np.pi * q * np.arange(k) / k for q in (1, 2, 3)]))]
    if semillas_extra is not None:
        bloques.append(("gnn", np.atleast_2d(semillas_extra)))
    origen = np.concatenate([[nombre] * len(b) for nombre, b in bloques])
    PHI = np.vstack([b for _, b in bloques])
    paso = 0.9 / (P * x[None, :]).sum(1).max()
    for _ in range(ITER_DESCENSO_PESADO):
        Fm = fuerza_pesada(x, P, PHI)
        if np.abs(Fm).max() < 1e-8:
            break
        PHI = PHI + paso * Fm
    mejor, espurios_aleatorios = None, 0
    for i, phi in enumerate(PHI):
        for _ in range(30):
            Fv = fuerza_pesada(x, P, phi[None])[0]
            if np.abs(Fv).max() < 1e-13:
                break
            phi = phi + np.linalg.lstsq(jacobiano_pesado(x, P, phi), -Fv, rcond=None)[0]
        phi = np.mod(phi - phi[0], 2 * np.pi)
        q = cantidades(x, phi, P)
        if q["F"] < 1e-9 and q["Dmin"] >= DELTA_ETIQUETA and q["lam2"] >= DELTA_ETIQUETA and q["r"] <= R_MAX:
            espurios_aleatorios += origen[i] == "aleatoria"
            margen = min(q["Dmin"], q["lam2"])
            if mejor is None or margen > mejor["margen"]:
                mejor = dict(margen=margen, theta=phi.copy(), lam2=q["lam2"], Dmin=q["Dmin"], r=q["r"], origen=origen[i])
    if mejor is None:
        return dict(etiqueta="sync_emp", theta=None, lam2=None, Dmin=None, margen=None, r=None, origen=None,
                    cuenca=0.0)
    return dict(etiqueta="no_sync_cert", cuenca=espurios_aleatorios / arranques, **mejor)


def mu_sistema(x, P):
    return float((P @ x).min())


def sistema_sdp_sintetico(rng_l, delta=None):
    """Semilla sintética con las densidades del SDP en un margen δ (o None si es infactible).
    El SDP deja D_a y λ₂ justo en δ, así que con δ = DELTA_ETIQUETA el sistema queda en el borde y el redondeo
    numérico lo puede etiquetar mal. Por eso δ se elige log-uniforme en `DELTAS_SDP_DATASET`, por encima de
    DELTA_ETIQUETA: así hay sistemas con más o menos margen (y, al perturbarlos, positivos y negativos)."""
    for _ in range(20):
        r = semilla_sintetica(rng_l)
        if r is None or not (K_PESADO[0] <= len(r[0]) <= K_PESADO[1]):
            continue
        x, phi = r
        d_ = delta if delta is not None else float(np.exp(rng_l.uniform(*np.log(DELTAS_SDP_DATASET))))
        mu, P = SDPClusters(len(x), d_).resolver(x, phi)
        if P is not None and factible(cantidades(x, phi, P), d_):
            return x, phi, P
    return None


def generar_sistema(familia, rng_l):
    """Devuelve (x, P, φ semilla o None) de la familia pedida."""
    if familia in ("sdp_sintetica", "perturbada"):
        r = sistema_sdp_sintetico(rng_l)
        if r is None:
            return None
        x, phi, P = r
        if familia == "perturbada":
            sigma = float(np.exp(rng_l.uniform(np.log(0.003), np.log(0.1))))
            R = np.triu(rng_l.normal(0, sigma, P.shape), 1)
            P = np.clip(P + R + R.T, 0, 1); np.fill_diagonal(P, 1.0)
            x = ajustar_tamanos(x * np.exp(rng_l.normal(0, sigma, len(x))))
        return x, P, phi
    k = int(rng_l.integers(K_PESADO[0], K_PESADO[1] + 1))
    x = ajustar_tamanos(rng_l.dirichlet(np.full(k, rng_l.uniform(1, 10))))
    if familia == "aleatoria":
        P = rng_l.uniform(0, 1, (k, k)) ** rng_l.uniform(0.3, 3)
    elif familia == "anillo":
        perfil = np.sort(rng_l.uniform(0, 1, k // 2 + 1))[::-1]; perfil[0] = 1.0
        d = np.abs(np.arange(k)[:, None] - np.arange(k)[None, :]); d = np.minimum(d, k - d)
        P = perfil[d] + rng_l.normal(0, 0.05, (k, k))
    else:
        raise ValueError(familia)
    P = np.triu(P, 1); P = np.clip(P + P.T, 0, 1); np.fill_diagonal(P, 1.0)
    return x, P, None


def registro_pesado(familia, semilla):
    warnings.filterwarnings("ignore")
    rng_l = np.random.default_rng(semilla)
    r = generar_sistema(familia, rng_l)
    if r is None:
        return None
    x, P, phi = r
    extra = None if phi is None else phi[None, :]
    return dict(x=x, P=P, n=len(x), familia=familia, mu=mu_sistema(x, P), **etiquetar_pesado(x, P, rng_l, extra))


def entrada_pesada(x, P, rng_local):
    k = len(x)
    W = P * x[None, :]
    grado = W.sum(1)
    w = np.outer(x, x) * P; np.fill_diagonal(w, 0)
    Dm = 1 / np.sqrt(w.sum(1) + 1e-12)
    _, V = np.linalg.eigh(np.eye(k) - Dm[:, None] * w * Dm[None, :])
    pe = np.zeros((k, K_PE))
    kk = min(K_PE, k - 1)
    pe[:, :kk] = V[:, 1:kk + 1] * rng_local.choice([-1, 1], size=kk)
    rasgos_v = np.hstack([k * x[:, None], grado[:, None], grado[:, None] / grado.max(), pe,
                          rng_local.normal(size=(k, K_AZAR))])
    return W / grado[:, None], rasgos_v


D_IN_PESADA = 3 + K_PE + K_AZAR


def lote_pesado(regs, rng_local, con_fases=True):
    n_max = max(r["n"] for r in regs)
    B = len(regs)
    A = np.zeros((B, n_max, n_max)); Xv = np.zeros((B, n_max, D_IN_PESADA))
    M = np.zeros((B, n_max)); TH = np.zeros((B, n_max)); tiene = np.zeros(B)
    for b, r in enumerate(regs):
        a, xv = entrada_pesada(r["x"], r["P"], rng_local)
        n = r["n"]
        A[b, :n, :n], Xv[b, :n], M[b, :n] = a, xv, 1
        if con_fases and r.get("theta") is not None:
            TH[b, :n], tiene[b] = r["theta"], 1
    t = lambda z: torch.tensor(z, dtype=torch.float32)
    return t(A), t(Xv), t(M), t(TH), t(tiene), t([etiqueta_binaria(r) if "etiqueta" in r else 0 for r in regs])


hallazgos_pesados = []
if GNN_PESADA and tiempo_agotado():
    log("TIEMPO: no se corre la GNN pesada")
elif GNN_PESADA:
    # --- control del etiquetador: una semilla sintética con su SDP tiene que dar positivo ---
    t0 = time.time()
    r_ctrl = sistema_sdp_sintetico(np.random.default_rng(SEMILLA + 7), delta=1e-3)
    e_ctrl = etiquetar_pesado(r_ctrl[0], r_ctrl[2], np.random.default_rng(1), r_ctrl[1][None, :])
    e_sync = etiquetar_pesado(np.full(5, 0.2), np.ones((5, 5)), np.random.default_rng(1))
    log(f"control etiquetador pesado: sintética+SDP → {e_ctrl['etiqueta']} (origen {e_ctrl['origen']}), "
        f"completo k=5 → {e_sync['etiqueta']}   (esperado: no_sync_cert y sync_emp)  {time.time() - t0:.1f} s")

    # --- dataset equilibrado ---
    FAMILIAS_PESADAS = ["sdp_sintetica", "perturbada", "aleatoria", "anillo"]
    cupo_p = N_SISTEMAS_PESADOS // len(FAMILIAS_PESADAS)
    medio_p = cupo_p // 2
    dataset_pesado = []
    t0 = time.time()
    for familia in FAMILIAS_PESADAS:
        pos, neg, intentos = [], [], 0
        while (len(pos) < medio_p or len(neg) < medio_p) and intentos < INTENTOS_POR_FAMILIA * cupo_p:
            tanda = Parallel(n_jobs=N_JOBS)(delayed(registro_pesado)(familia, int(rng.integers(1e9)))
                                            for _ in range(min(64, cupo_p)))
            intentos += len(tanda)
            for r in tanda:
                if r is not None:
                    (pos if es_positivo(r) else neg).append(r)
        elegidos = pos[:medio_p] + neg[:medio_p]
        dataset_pesado += elegidos
        log(f"  sistemas {familia:14s} {len(elegidos):4d}, {sum(es_positivo(r) for r in elegidos):4d} positivos "
            f"(de {intentos} etiquetados)  ({time.time() - t0:.0f} s)")
    guardar(dataset_pesado, "dataset_pesado.pkl")

    # --- entrenamiento y evaluación por familia ---
    y_p = np.array([etiqueta_binaria(r) for r in dataset_pesado])
    orden_p = rng.permutation(len(dataset_pesado))
    corte_p = int(0.75 * len(orden_p))
    idx_ent_p, idx_pru_p = orden_p[:corte_p], orden_p[corte_p:]
    gnn_p = GNNKuramoto(d_in=D_IN_PESADA)
    base_sd = torch.load(os.path.join(DATOS, "gnn_base.pt"))
    gnn_p.load_state_dict({k_: v for k_, v in base_sd.items() if not k_.startswith("entrada.")}, strict=False)
    t0 = time.time()
    entrenar(gnn_p, idx_ent_p, EPOCAS_GNN_PESADA, datos=dataset_pesado, armar_lote=lote_pesado)
    log(f"GNN pesada: entrenamiento {time.time() - t0:.0f} s")
    prob_p, _ = predecir(gnn_p, dataset_pesado, armar_lote=lote_pesado)
    log(f"GNN pesada en prueba: {metricas(y_p[idx_pru_p], prob_p[idx_pru_p])}")
    tabla_fam_p = pd.DataFrame(metricas_por_familia(dataset_pesado, idx_pru_p, prob_p, "GNN pesada"))
    log("GNN pesada por familia (prueba):\n" + tabla_fam_p.to_string(index=False))
    tabla_fam_p.to_csv(os.path.join(RES, "gnn_pesada_por_familia.csv"), index=False)
    entrenar(gnn_p, np.arange(len(dataset_pesado)), max(1, EPOCAS_GNN_PESADA // 2), datos=dataset_pesado,
             armar_lote=lote_pesado)
    torch.save(gnn_p.state_dict(), os.path.join(RES, "gnn_pesada.pt"))

    # --- annealing continuo en (x, P) ---
    def mover_sistema(x, P, rng_l):
        k = len(x); u = rng_l.random()
        if u < 0.7:
            a, b = rng_l.choice(k, 2, replace=False)
            P = P.copy(); P[a, b] = P[b, a] = np.clip(P[a, b] + rng_l.normal(0, PASO_P_SA), 0, 1)
        elif u < 0.9 or k >= K_MAX_SEMILLA:
            x = ajustar_tamanos(x * np.exp(rng_l.normal(0, PASO_X_SA, k)))
        else:                                       # partir un cluster en dos mitades conectadas entre sí
            a = int(rng_l.integers(k))
            x = np.append(x, x[a] / 2); x[a] = x[-1]; x = ajustar_tamanos(x)
            P = np.pad(P, ((0, 1), (0, 1))); P[-1, :-1] = P[a, :-1]; P[:-1, -1] = P[:-1, a]; P[-1, -1] = 1.0
            P[a, -1] = P[-1, a] = 1.0
        return x, P

    def recocido_pesado(x, P, familia, estado, semilla):
        warnings.filterwarnings("ignore")
        torch.set_num_threads(1)
        modelo = GNNKuramoto(d_in=D_IN_PESADA); modelo.load_state_dict(estado)
        rng_l = np.random.default_rng(semilla)
        prob = lambda x, P: predecir(modelo, [dict(x=x, P=P, n=len(x))], armar_lote=lote_pesado)
        p, f = prob(x, P); p, fase = p[0], f[0]
        puntaje = lambda x, P, p: PESO_MU_SA_PESADO * mu_sistema(x, P) + BETA_SA * np.log(p + 1e-6)
        actual = puntaje(x, P, p)
        hallazgos, llamadas, ultimo, cambio = [], 0, -VERIFICAR_CADA, True
        for paso_ in range(PASOS_SA_PESADO + 1):
            if paso_ > 0:
                xn, Pn = mover_sistema(x, P, rng_l)
                pn, fn = prob(xn, Pn); pn, fn = pn[0], fn[0]
                nuevo = puntaje(xn, Pn, pn)
                if nuevo > actual or rng_l.random() < np.exp((nuevo - actual) / TEMPERATURA_SA_PESADO):
                    x, P, actual, p, fase, cambio = xn, Pn, nuevo, pn, fn, True
            if p > UMBRAL_GNN and cambio and paso_ - ultimo >= VERIFICAR_CADA:
                ultimo, cambio = paso_, False
                llamadas += 1
                r = etiquetar_pesado(x, P, rng_l, semillas_extra=np.vstack([fase, -fase]))
                if r["etiqueta"] == "no_sync_cert":
                    hallazgos.append(dict(x=x.copy(), phi=r["theta"], P=P.copy(), familia=familia, mu=mu_sistema(x, P),
                                          p_gnn=p, **{k_: v for k_, v in r.items() if k_ != "theta"}))
        return hallazgos, llamadas

    def inicio_sa_pesado(familia, rng_l):
        if familia == "positivos_pesados":
            pos = sorted([r for r in dataset_pesado if es_positivo(r)], key=lambda r: -r["mu"])
            if pos:
                r = pos[int(rng_l.integers(min(len(pos), 20)))]
                return r["x"].copy(), r["P"].copy()
            familia = "sdp_sintetica"
        if familia == "sdp_sintetica":
            r = sistema_sdp_sintetico(rng_l)
            if r is not None:
                return r[0], r[2]
        x, P, _ = generar_sistema("aleatoria", rng_l)
        return x, P

    FAMILIAS_SA_PESADO = ["positivos_pesados", "sdp_sintetica", "aleatoria"]
    trabajos = [(*inicio_sa_pesado(fam, rng), fam, int(rng.integers(1e9)))
                for fam in FAMILIAS_SA_PESADO for _ in range(CORRIDAS_SA_PESADO)]
    estado_p = {k_: v.clone() for k_, v in gnn_p.state_dict().items()}
    t0 = time.time()
    salidas = Parallel(n_jobs=N_JOBS)(delayed(recocido_pesado)(x, P, fam, estado_p, s_) for x, P, fam, s_ in trabajos)
    crudos = sorted([h for hs, _ in salidas for h in hs], key=lambda h: -h["mu"])
    llamadas = sum(c for _, c in salidas)
    firma_pesada = lambda x, phi: np.array([np.abs(np.sum(x * np.exp(1j * l * phi))) for l in range(1, 7)])
    for h in crudos:                                # sin repetir: mismo k y firmas cercanas = el mismo patrón
        f_ = firma_pesada(h["x"], h["phi"])
        if any(len(g_["x"]) == len(h["x"]) and np.linalg.norm(g_["firma"] - f_) < TOL_DIVERSIDAD for g_ in hallazgos_pesados):
            continue
        hallazgos_pesados.append(dict(h, firma=f_))
    log(f"annealing pesado: {len(trabajos)} corridas, {llamadas} llamadas al etiquetador, {len(crudos)} positivos, "
        f"{len(hallazgos_pesados)} sin repetir ({time.time() - t0:.0f} s)")
    if hallazgos_pesados:
        log("\n" + pd.DataFrame([dict(familia=h["familia"], k=len(h["x"]), mu=round(h["mu"], 5), p_gnn=round(h["p_gnn"], 3),
                                      margen=f"{h['margen']:.1e}", twisted=es_twisted_aprox(h["x"], h["phi"]))
                                 for h in hallazgos_pesados[:15]]).to_string(index=False))
    guardar(hallazgos_pesados, "hallazgos_pesados.pkl")

# %% [markdown]
# ### Semillas, sistemas de la GNN pesada y patadas → TOPE
#
# Las diez familias se generan y se seleccionan por separado (SDP en `DELTA_INICIO_SEMILLAS`, pre-pulido de las
# `FACTOR_PRESELECCION` × `TOP_POR_FAMILIA` mejores, entran las `TOP_POR_FAMILIA` mejores de cada una). Los sistemas
# de la GNN pesada que son twisted se patean, igual que los twisted de los grafos.

# %%
assert DELTA_INICIO_SEMILLAS in DELTAS, "DELTA_INICIO_SEMILLAS tiene que ser uno de los DELTAS"
pesadas = []
for h in hallazgos_pesados:
    if TRATAR_TWISTED != "nada" and es_twisted_aprox(h["x"], h["phi"]):
        if TRATAR_TWISTED == "patada":
            fuentes_patada.append(dict(padre="gnn_pesada", x=h["x"].copy(), phi=h["phi"].copy(),
                                       etapa="twisted de la GNN pesada"))
        continue
    pesadas.append(dict(tipo="pesada", padre=None, x=h["x"].copy(), phi=h["phi"].copy(), etapa="GNN pesada + annealing",
                        x_gnn=h["x"].copy(), phi_gnn=h["phi"].copy(), P_gnn=h["P"].copy(), mu_gnn=h["mu"],
                        familia_sa=h["familia"]))
log(f"GNN pesada: {len(hallazgos_pesados)} sistemas propuestos, {len(pesadas)} no twisted")
nivel_semillas = DELTAS.index(DELTA_INICIO_SEMILLAS)
rng_sem = np.random.default_rng(SEMILLA + 1)
t0 = time.time()

# --- diez familias, cada una seleccionada por separado ---
sinteticas, top_s, mus_s, filas_fam, mus_por_familia = [], [], [], [], {}
if SEMILLAS_SINTETICAS:
    for fam in FAMILIAS_SEMILLAS:
        t1 = time.time()
        lista, intentos_s = [], 0
        while len(lista) < N_POR_FAMILIA and intentos_s < 20 * N_POR_FAMILIA:
            intentos_s += 1
            r = semilla_familia(fam, rng_sem)
            if r is not None:
                lista.append(dict(tipo="sintetica", geometria=fam, padre=None, x=r[0], phi=r[1], etapa=f"receta {fam}"))
        top_f, mus_f = seleccionar(lista, TOP_POR_FAMILIA, DELTA_INICIO_SEMILLAS)
        sinteticas += lista; top_s += top_f; mus_s += mus_f; mus_por_familia[fam] = mus_f
        filas_fam.append(dict(geometria=fam, generadas=len(lista), factibles=len(mus_f),
                              k_medio=round(float(np.mean([len(s_["x"]) for s_ in lista])), 1) if lista else np.nan,
                              mu_crudo_max=round(max(mus_f, default=np.nan), 5),
                              mu_prepulido_max=round(max([s_["mu0"] for s_ in top_f], default=np.nan), 5),
                              mu_prepulido_min_entra=round(min([s_["mu0"] for s_ in top_f], default=np.nan), 5),
                              entran=len(top_f), segundos=round(time.time() - t1)))
        log(f"  familia {fam:20s}: {len(mus_f)}/{len(lista)} factibles, μ crudo máx {filas_fam[-1]['mu_crudo_max']}, "
            f"entran {len(top_f)} con μ pre-pulido {filas_fam[-1]['mu_prepulido_min_entra']}–{filas_fam[-1]['mu_prepulido_max']} "
            f"({filas_fam[-1]['segundos']} s)")
    tabla_familias_semillas = pd.DataFrame(filas_fam)
    log("\n" + tabla_familias_semillas.to_string(index=False))
    tabla_familias_semillas.to_csv(os.path.join(RES, "familias_semillas.csv"), index=False)
pateadas = []
for f_ in fuentes_patada:
    for _ in range(PATADAS_POR_TWISTED):
        for _ in range(20):
            r = patada(f_["x"], f_["phi"], rng_sem)
            if r is not None:
                pateadas.append(dict(tipo="patada", padre=f_["padre"], x=r[0], phi=r[1], etapa=f_["etapa"],
                                     x_padre=f_["x"], phi_padre=f_["phi"]))
                break
log(f"SEMILLAS: {len(sinteticas)} sintéticas y {len(pateadas)} patadas (de {len(fuentes_patada)} twisted)")

top_p, mus_p = seleccionar(pateadas, TOP_PATADAS, DELTA_INICIO_SEMILLAS)
top_g, mus_g = seleccionar(pesadas, TOP_PESADAS, DELTA_INICIO_SEMILLAS)
log(f"  GNN pesada: factibles {len(mus_g)}/{len(pesadas)}, entran {len(top_g)} (μ "
    f"{min([s_['mu0'] for s_ in top_g], default=np.nan):.5f}–{max([s_['mu0'] for s_ in top_g], default=np.nan):.5f})")
log(f"SDP de las semillas en δ={DELTA_INICIO_SEMILLAS:g} ({time.time() - t0:.0f} s): "
    f"sintéticas factibles {len(mus_s)}/{len(sinteticas)}, patadas factibles {len(mus_p)}/{len(pateadas)}")
log(f"  entran a TOPE: {len(top_s)} sintéticas (μ {min([s_['mu0'] for s_ in top_s], default=np.nan):.5f}–"
    f"{max([s_['mu0'] for s_ in top_s], default=np.nan):.5f}) y {len(top_p)} patadas (μ "
    f"{min([s_['mu0'] for s_ in top_p], default=np.nan):.5f}–{max([s_['mu0'] for s_ in top_p], default=np.nan):.5f})")

semillas_info = {}
for pref, lista in (("s", top_s), ("p", top_p), ("g", top_g)):
    for j, s_ in enumerate(lista):
        sid = f"{pref}{j:03d}"
        s_["id"] = sid
        semillas_info[sid] = s_
        estados[sid] = dict(id=sid, familia={"s": "sintetica", "p": "patada", "g": "gnn_pesada"}[pref], k=len(s_["x"]),
                            z=codificar(s_["x"], s_["phi"]), mu_sdp1=s_["mu0"], niveles=[], presupuesto=GENS_BASE,
                            estado="activo", alarma=False, motivo=None, nivel_inicio=nivel_semillas,
                            padre=s_["padre"], geometria=s_.get("geometria"))

fig, ax = plt.subplots(1, 3, figsize=(15, 3.4), sharex=True)
for a, mus_, tit in ((ax[0], mus_s, "sintéticas"), (ax[1], mus_p, "patadas a twisted"), (ax[2], mus_g, "GNN pesada")):
    if mus_:
        a.hist(mus_, bins=30, color="C2")
    a.set_title(f"μ del SDP crudo en δ={DELTA_INICIO_SEMILLAS:g}: {tit} ({len(mus_)} factibles)", fontsize=10)
    lineas_cotas(a, mus_); a.set_xlabel("μ")
plt.tight_layout(); plt.savefig(os.path.join(RES, "hist_semillas.png"), dpi=120); plt.show()
if mus_por_familia:                                  # μ del SDP crudo por geometría
    fig, a = plt.subplots(figsize=(12, 3.8))
    a.boxplot([mus_por_familia[f_] if mus_por_familia[f_] else [np.nan] for f_ in FAMILIAS_SEMILLAS], showfliers=False)
    a.set_xticks(range(1, len(FAMILIAS_SEMILLAS) + 1)); a.set_xticklabels(FAMILIAS_SEMILLAS, rotation=30, ha="right", fontsize=8)
    a.axhline(M_OBJETIVO, color="tab:red", lw=1); a.set_ylabel(f"μ del SDP crudo (δ={DELTA_INICIO_SEMILLAS:g})")
    a.set_title("Semillas factibles por geometría")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "familias_semillas.png"), dpi=120); plt.show()
guardar(dict(sinteticas=sinteticas, pateadas=pateadas, pesadas=pesadas, fuentes_patada=fuentes_patada), "semillas.pkl")

if semillas_info:
    log(f"\nTOPE para las semillas, desde δ={DELTA_INICIO_SEMILLAS:g}")
    finales += ejecutar_niveles(list(semillas_info), nivel_semillas, patear=False)

# %% [markdown]
# ## 6d. Rondas evolutivas
#
# Después del TOPE de las semillas, se eligen las `PADRES_EVOLUCION` mejores semillas y candidatos (no controles, no
# convergidos a twisted), ordenados por el nivel de δ más chico que alcanzaron y, a igual nivel, por μ. Así entran
# también los descartados que llegaron lejos. Cada padre, en su último nivel
# alcanzado, genera `HIJOS_POR_PADRE` mutaciones:
# - tamaños × e^{N(0, 0.1)} y fases + N(0, 0.05);
# - con probabilidad 0.3 se parte un cluster en dos (fases ±ε, ε en `EPS_PARTICION`, reparto al azar del tamaño);
# - con probabilidad 0.2 se juntan los dos clusters de fases más cercanas.
#
# Los hijos pasan por la misma preselección que las semillas (SDP en `DELTA_INICIO_SEMILLAS` y pulido corto) y entran a
# TOPE los `TOP_EVOLUCION` mejores. Se repite `RONDAS_EVOLUTIVAS` veces, sin repetir padres, mientras haya tiempo.
#
# Además de las `PADRES_EVOLUCION` mejores, es padre la mejor semilla de cada geometría
# (`FAMILIAS_SEMILLAS`) que no esté ya representada, así cada geometría tiene al menos un linaje mutado.

# %%
def mutar(x, phi, rng_s):
    x = x * np.exp(rng_s.normal(0, 0.1, len(x)))
    phi = phi + rng_s.normal(0, 0.05, len(x))
    u = rng_s.random()
    if u < 0.3 and len(x) < K_MAX_SEMILLA:                       # partir un cluster
        a = int(rng_s.integers(len(x))); eps = rng_s.uniform(*EPS_PARTICION); f = rng_s.uniform(0.3, 0.7)
        x = np.append(x, x[a] * (1 - f)); x[a] *= f
        phi = np.append(phi, phi[a] + eps); phi[a] -= eps
    elif u < 0.5 and len(x) > 3:                                 # juntar los dos de fases más cercanas
        o = np.argsort(np.mod(phi, 2 * np.pi)); a_ = np.mod(phi[o], 2 * np.pi)
        j = int(np.argmin(np.diff(np.append(a_, a_[0] + 2 * np.pi))))
        a, b = o[j], o[(j + 1) % len(o)]
        phi[a] = np.angle(x[a] * np.exp(1j * phi[a]) + x[b] * np.exp(1j * phi[b])); x[a] += x[b]
        x, phi = np.delete(x, b), np.delete(phi, b)
    return cerrar_semilla(x, phi)


def avance(e):
    val = [(e["nivel_inicio"] + j, u["mu"]) for j, u in enumerate(e["niveles"]) if u["P"] is not None and np.isfinite(u["mu"])]
    return max(val) if val else (-1, -1.0)


evolucion = []
usados = set()
for ronda in range(RONDAS_EVOLUTIVAS):
    if tiempo_agotado():
        log(f"TIEMPO: pasaron más de {TIEMPO_MAX_HORAS} h; no se corre la ronda evolutiva {ronda}")
        break
    posibles = [e for e in estados.values() if not e["familia"].startswith("control") and e["id"] not in usados
                and avance(e)[0] >= 0 and "twisted" not in (e["motivo"] or "")]
    padres = sorted(posibles, key=avance, reverse=True)[:PADRES_EVOLUCION]
    for fam in FAMILIAS_SEMILLAS:                  # la mejor de cada geometría que no esté representada
        de_fam = [e for e in posibles if e.get("geometria") == fam]
        if de_fam and not any(p_.get("geometria") == fam for p_ in padres):
            padres.append(max(de_fam, key=avance))
    if not padres:
        break
    usados |= {e["id"] for e in padres}
    hijos = []
    for e in padres:
        u = [u for u in e["niveles"] if u["P"] is not None and np.isfinite(u["mu"])][-1]
        for _ in range(HIJOS_POR_PADRE):
            for _ in range(20):
                r = mutar(u["x"].copy(), u["phi"].copy(), rng_sem)
                if r is not None:
                    hijos.append(dict(tipo="evolucion", padre=e["id"], x=r[0], phi=r[1],
                                      etapa=f"μ={u['mu']:.5f}, δ={u['delta']:g}", x_padre=u["x"], phi_padre=u["phi"],
                                      P_padre=u["P"], mu_padre=u["mu"]))
                    break
    t0 = time.time()
    top_e, mus_e = seleccionar(hijos, TOP_EVOLUCION, DELTA_INICIO_SEMILLAS)
    log(f"\nEVOLUCIÓN ronda {ronda}: padres {[(e['id'], e.get('geometria'), round(avance(e)[1], 5)) for e in padres]}; "
        f"{len(hijos)} hijos, {len(mus_e)} factibles; entran {len(top_e)} (μ pre-pulido "
        f"{min([s_['mu0'] for s_ in top_e], default=np.nan):.5f}–{max([s_['mu0'] for s_ in top_e], default=np.nan):.5f}) "
        f"({time.time() - t0:.0f} s)")
    ids = []
    for j, s_ in enumerate(top_e):
        sid = f"e{ronda}{j:02d}"
        s_["id"] = sid
        semillas_info[sid] = s_
        estados[sid] = dict(id=sid, familia="evolucion", k=len(s_["x"]), z=codificar(s_["x"], s_["phi"]),
                            mu_sdp1=s_["mu0"], niveles=[], presupuesto=GENS_BASE, estado="activo", alarma=False,
                            motivo=None, nivel_inicio=nivel_semillas, padre=s_["padre"],
                            geometria=estados[s_["padre"]].get("geometria"))
        ids.append(sid)
    evolucion.append(dict(ronda=ronda, padres=[e["id"] for e in padres], hijos=hijos, entran=ids))
    if ids:
        finales += ejecutar_niveles(ids, nivel_semillas, patear=False)
guardar(evolucion, "evolucion.pkl")
log(f"Tiempo total hasta acá: {(time.time() - T_INICIO) / 3600:.2f} h")

# %%
mejor_final = max([estados[i]["niveles"][-1]["mu"] for i in finales], default=-1)
if RESCATE and mejor_final < M_OBJETIVO and en_espera:
    log(f"\nRESCATE: ningún final supera {M_OBJETIVO}; se retoman {len(en_espera)} en espera")
    for i in list(en_espera):
        estados[i]["estado"] = "activo"
        nivel_i = estados[i]["nivel_inicio"] + len(estados[i]["niveles"])
        finales += ejecutar_niveles([i], nivel_i, solo_techo=True, patear=False)

log(f"\nRESUMEN TOPE: {len(finales)} llegaron al final {finales}; {len(descartados)} descartados; "
    f"{sum(e['estado'] == 'espera' for e in estados.values())} quedaron en espera")
c_ok = estados.get("control_okeeffe_k7")
if c_ok:
    log("CONTROL O'Keeffe: " + ", ".join(f"δ={n_['delta']:g}: μ={n_['mu']:.5f}" for n_ in c_ok["niveles"]) +
        "   (referencia del informe: 0.6821 / 0.6879 / 0.6906 para δ = 1e-4..1e-6; prueba con pulido: 0.6868 / 0.6894 / 0.6907 / 0.6919)")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
antes = [e["mu_sdp1"] for e in estados.values()]
despues = [e["niveles"][-1]["mu"] for e in estados.values() if e["niveles"] and np.isfinite(e["niveles"][-1]["mu"])]
bins = np.linspace(min(antes + despues) - 0.01, max(antes + despues + [M_OBJETIVO]) + 0.01, 30)
ax[0].hist(antes, bins=bins); ax[0].set_title("μ antes de CMA-ES + SDP (primer SDP)")
ax[1].hist(despues, bins=bins, color="C1"); ax[1].set_title("μ en el último nivel alcanzado")
lineas_cotas(ax[0], antes); lineas_cotas(ax[1], despues)
for a in ax:
    a.set_xlabel("μ"); a.set_ylabel("candidatos")
plt.tight_layout(); plt.savefig(os.path.join(RES, "hist_tope.png"), dpi=120); plt.show()

# %% [markdown]
# ### Sistemas finales: geometría y densidades óptimas
#
# Izquierda: cada cluster en su fase sobre el círculo; el área y el color del punto indican el tamaño x_a
# (escala `plasma`). Las aristas tienen opacidad ∝ p_ab, azules si cos(φ_a − φ_b) > 0 (estabilizan) y rojas si < 0.
# Derecha: la matriz P ordenada por fase y, debajo, la franja k×1 con los tamaños x_a en el mismo orden y la
# misma escala de color que los puntos.

# %%
def dibujar_sistema(x, phi, P, titulo, ax_c, ax_m, barra_P=True):
    """Izquierda: clusters en su fase; color y área del punto según el tamaño x_a.
    Derecha: P ordenada por fase y, debajo, la franja k×1 con los tamaños x_a en el mismo orden y colores."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    orden = np.argsort(np.mod(phi, 2 * np.pi))
    k = len(x)
    # escala entre el menor y el mayor tamaño del sistema (si son todos iguales, un único color intermedio)
    norma = plt.Normalize(x.min() - 1e-12, x.max()) if np.ptp(x) > 1e-4 else plt.Normalize(x.min() - 1, x.min() + 1)
    ax_c.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.85"))
    for a in range(k):
        for b in range(a + 1, k):
            if P[a, b] > 0.02:
                col = "tab:blue" if np.cos(phi[a] - phi[b]) > 0 else "tab:red"
                ax_c.plot([np.cos(phi[a]), np.cos(phi[b])], [np.sin(phi[a]), np.sin(phi[b])], color=col,
                          alpha=0.12 + 0.55 * P[a, b], lw=0.4 + 2.0 * P[a, b] * min(1, 7 / k))
    ax_c.scatter(np.cos(phi), np.sin(phi), s=1800 * x, c=x, cmap="plasma", norm=norma, zorder=3,
                 edgecolors="k", linewidths=0.6)
    ax_c.set_xlim(-1.3, 1.3); ax_c.set_ylim(-1.3, 1.3)
    ax_c.set_aspect("equal"); ax_c.axis("off"); ax_c.set_title(titulo, fontsize=9)
    im = ax_m.imshow(P[np.ix_(orden, orden)], vmin=0, vmax=1, cmap="viridis")
    ax_m.set_xticks([]); ax_m.set_yticks([])
    ax_m.set_title("P ordenada por fase", fontsize=9)
    div = make_axes_locatable(ax_m)
    ax_x = div.append_axes("bottom", size=0.22, pad=0.06)
    ax_x.imshow(x[orden][None, :], cmap="plasma", norm=norma, aspect="auto")
    ax_x.set_yticks([0]); ax_x.set_yticklabels(["x"], fontsize=8); ax_x.set_xticks([])
    if k <= 8:
        for j, v in enumerate(x[orden]):
            ax_x.text(j, 0, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                      color="white" if norma(v) < 0.5 else "black")
    if barra_P:
        plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))
    return im

if finales:
    fig, axs = plt.subplots(len(finales), 2, figsize=(8, 4 * len(finales)), squeeze=False)
    for fila, i in zip(axs, finales):
        u = estados[i]["niveles"][-1]
        dibujar_sistema(u["x"], u["phi"], u["P"], f"{i}  μ={u['mu']:.5f}  δ={u['delta']:g}", fila[0], fila[1])
    plt.tight_layout(); plt.savefig(os.path.join(RES, "finales.png"), dpi=120); plt.show()

# %% [markdown]
# ## 8. Compresión
#
# Se juntan nodos con fases casi iguales. Para no juntar separaciones reales (por ejemplo las mitades de
# O'Keeffe, a ~0.03 rad), se busca un **hueco de escalas** en las separaciones entre fases consecutivas: el
# mayor salto (≥ `RATIO_HUECO`) entre una separación y la siguiente, por debajo de `TAU_MAX`. Se juntan las que
# quedan del lado chico. Al cluster nuevo se le asigna x_c = Σ x_a, φ_c = promedio circular ponderado y
# densidades promedio $p_{cd}=\sum_{a\in c,b\in d}x_ax_bp_{ab}/(X_cX_d)$.
#
# Después se **re-resuelve** el SDP en el sistema comprimido y, si hace falta, se pule con un CMA-ES corto
# seguido del pulido convexo secuencial (con `PULIDO_SCP`).
# Si μ cae más de `TOL_CAIDA`, se prueba un umbral de compresión más chico, y si no alcanza, no se comprime.
# Como diagnóstico se informa la dispersión de las filas de P dentro de cada grupo.
#
# La validez no depende del umbral: lo que se certifica después es el sistema comprimido re-resuelto.

# %%
def huecos(phi):
    orden = np.argsort(np.mod(phi, 2 * np.pi))
    a = np.mod(phi, 2 * np.pi)[orden]
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    return orden, gaps


def umbrales_candidatos(phi):
    _, gaps = huecos(phi)
    g = np.sort(gaps[gaps > 0])
    opciones = []
    for j in range(len(g) - 1):
        if g[j] <= TAU_MAX and g[j + 1] / max(g[j], 1e-300) >= RATIO_HUECO:
            opciones.append((g[j + 1] / max(g[j], 1e-300), np.sqrt(g[j] * g[j + 1])))
    return [tau for _, tau in sorted(opciones, reverse=True)]


def agrupar(phi, tau):
    orden, gaps = huecos(phi)
    grupos, actual = [], [orden[0]]
    for j in range(1, len(orden)):
        if gaps[j - 1] < tau:
            actual.append(orden[j])
        else:
            grupos.append(actual); actual = [orden[j]]
    if grupos and gaps[-1] < tau:          # el último se une con el primero (círculo)
        grupos[0] = actual + grupos[0]
    else:
        grupos.append(actual)
    return grupos


def fusionar(x, phi, P, grupos):
    X = np.array([x[g].sum() for g in grupos])
    Phi = np.array([np.angle(np.sum(x[g] * np.exp(1j * phi[g]))) for g in grupos])
    Pc = np.array([[np.sum(np.outer(x[g], x[h]) * P[np.ix_(g, h)]) / (X[a] * X[b]) for b, h in enumerate(grupos)]
                   for a, g in enumerate(grupos)])
    dispersion = max((np.abs(P[g] - P[g].mean(0)).max() for g in grupos if len(g) > 1), default=0.0)
    return X, Phi - Phi[0], Pc, dispersion


def comprimir(u, semilla):
    x, phi, P, delta, mu = u["x"], u["phi"], u["P"], u["delta"], u["mu"]
    intentos = []
    for tau in umbrales_candidatos(phi):
        grupos = agrupar(phi, tau)
        if len(grupos) == len(x):
            continue
        X, Phi, Pc, disp = fusionar(x, phi, P, grupos)
        k = len(X)
        mu_c, P_c = SDPClusters(k, delta).resolver(X, Phi)
        ok = mu_c is not None and factible(cantidades(X, Phi, P_c), delta)
        info = dict(tau=tau, k=k, mu_directo=mu_c if ok else np.nan, dispersion_filas=disp)
        if ok and mu_c >= mu - TOL_CAIDA:
            intentos.append(info)
            return dict(x=X, phi=Phi, P=P_c, mu=cantidades(X, Phi, P_c)["mu"], delta=delta, grupos=grupos,
                        k=k, pulido=False, intentos=intentos)
        pul = correr_nivel(dict(k=k, z=codificar(X, Phi)), delta, GENS_PULIDO, SIGMA_MIN * 10, semilla)
        if PULIDO_SCP and pul["P"] is not None:          # pulido convexo después del CMA-ES corto
            p_ = pulir(pul["x"], pul["phi"], delta)
            if p_ is not None and p_["mu"] > pul["mu"]:
                pul.update(x=p_["x"], phi=p_["phi"], P=p_["P"], mu=p_["mu"])
        info["mu_pulido"] = pul["mu"]
        intentos.append(info)
        if pul["P"] is not None and pul["mu"] >= mu - TOL_CAIDA:
            return dict(x=pul["x"], phi=pul["phi"], P=pul["P"], mu=pul["mu"], delta=delta, grupos=grupos, k=k,
                        pulido=True, intentos=intentos)
    return dict(x=x, phi=phi, P=P, mu=mu, delta=delta, grupos=[[i] for i in range(len(x))], k=len(x),
                pulido=False, intentos=intentos)


comprimidos = {}
for i in finales:
    for nivel, u in enumerate(estados[i]["niveles"]):
        if u["P"] is None:
            continue
        cmp_ = comprimir(u, int(rng.integers(1e9)))
        comprimidos[(i, nivel)] = cmp_
        log(f"{i} δ={u['delta']:g}: k {len(u['x'])} → {cmp_['k']}  μ {u['mu']:.6f} → {cmp_['mu']:.6f}  "
            f"pulido={cmp_['pulido']}  intentos={[{k_: (round(v, 6) if isinstance(v, float) else v) for k_, v in t.items()} for t in cmp_['intentos']]}")
guardar(comprimidos, "comprimidos.pkl")

# %% [markdown]
# ## 9. Certificación
#
# Para cada sistema comprimido (empezando por el δ más grande, que es el más robusto):
#
# 1. **Racionalizar.** Tamaños x_a ≈ σ_a/Σσ (enteros σ_a, Σσ ≤ `Q_MAX`), re-resolver el SDP con esos tamaños,
#    y redondear las densidades hacia abajo a q_ab/m. Se prueban `OPCIONES_TAMANOS` racionalizaciones de x y
#    todos los m de `M_DENOMINADORES`; se informa la de **mayor μ certificado** y, aparte, la mejor que todavía
#    se puede armar como grafo explícito.
# 2. **Camino simétrico** (como en tu notebook): si las fases son equiespaciadas y los tamaños iguales
#    (tolerancia `TOL_SIM`), se fijan φ_j = 2πj/k exactas, se simetriza P a circulante y se redondea. El
#    equilibrio vale exactamente por simetría, para cualquier redondeo.
# 3. **Camino general**: Newton en precisión de `DPS` dígitos para re-equilibrar las fases con los valores
#    redondeados, y **aritmética de intervalos**: test de Krawczyk (existencia y unicidad de un equilibrio
#    exacto en una caja de radio ~10⁻³⁰) y cotas inferiores rigurosas para λ₂ y min D_a en toda la caja (Weyl).
# 4. **Blow-up explícito** si m ≤ `M_MAX_BLOWUP`: cada cluster a se parte en σ_a subclusters de m vértices;
#    entre subclusters se ponen piezas circulantes de grado q_ab con saltos al azar. El grafo tiene N = m·Σσ
#    vértices y μ = min_a Σ_b q_ab σ_b /(m Σσ) exacto. Como todas las piezas son circulantes en Z_m, el
#    Hessiano completo se diagonaliza por bloques con Fourier: m problemas de Σσ × Σσ. Se chequea que haya
#    un solo autovalor nulo y el resto positivos. La semilla de los saltos se guarda, así el grafo se puede
#    reconstruir.
#
# **Qué es riguroso:** el paso 3 (sistema reducido) usa intervalos. El chequeo del paso 4 usa doble
# precisión con un margen muy superior al error de redondeo, pero no es una prueba con intervalos.

# %%
import mpmath
from mpmath import mp, iv
mp.dps = DPS
iv.dps = DPS


def opciones_tamanos(x, q_max=None):
    """Las OPCIONES_TAMANOS mejores aproximaciones x_a ≈ σ_a/Σσ con Σσ ≤ Q_MAX (menor error primero).
    Con Q_MAX grande se agregan las mejores con Σσ ≤ `Q_MAX_BLOWUP`, para que siga habiendo alguna
    que se pueda armar como grafo explícito."""
    if q_max is None and Q_MAX > Q_MAX_BLOWUP:
        a, b = opciones_tamanos(x, Q_MAX), opciones_tamanos(x, Q_MAX_BLOWUP)
        return a + [s_ for s_ in b if not any(np.array_equal(s_, t_) for t_ in a)]
    q_max = Q_MAX if q_max is None else q_max
    vistas, opciones = set(), []
    for S in range(len(x), q_max + 1):
        sig = np.maximum(1, np.round(x * S)).astype(int)
        while sig.sum() > S:
            sig[np.argmax(sig - x * S)] -= 1
        while sig.sum() < S:
            sig[np.argmin(sig - x * S)] += 1
        if sig.min() < 1 or tuple(sig) in vistas:
            continue
        vistas.add(tuple(sig))
        opciones.append((np.abs(sig / S - x).max(), S, sig))
    opciones.sort(key=lambda t: (t[0], t[1]))
    return [sig for _, _, sig in opciones[:OPCIONES_TAMANOS]]


def es_twisted(x, phi):
    k = len(x)
    _, gaps = huecos(phi)
    return k >= 3 and np.abs(gaps - 2 * np.pi / k).max() < TOL_SIM and np.abs(x - 1 / k).max() < TOL_SIM


def F_y_J(ctx, phi, x, P, k):
    """Equilibrio reducido (a = 1..k−1) y su jacobiano respecto de φ_1..φ_{k−1}, en el contexto mp o iv."""
    F = [ctx.mpf(0)] * (k - 1)
    J = ctx.matrix(k - 1, k - 1)
    for a in range(1, k):
        for b in range(k):
            if b == a or P[a][b] == 0:
                continue
            w = P[a][b] * x[b]
            F[a - 1] += w * ctx.sin(phi[b] - phi[a])
            c = w * ctx.cos(phi[b] - phi[a])
            if b > 0:
                J[a - 1, b - 1] += c
            J[a - 1, a - 1] -= c
    return F, J


def matriz_reducida(ctx, phi, x, P, k):
    Ms = ctx.matrix(k, k)
    D = []
    for a in range(k):
        Da = ctx.mpf(0)
        for b in range(k):
            c = ctx.cos(phi[a] - phi[b])
            Da += P[a][b] * x[b] * c
            if b != a:
                Ms[a, b] = -ctx.sqrt(x[a] * x[b]) * P[a][b] * c
                Ms[a, a] += x[b] * P[a][b] * c
        D.append(Da)
    return Ms, D


def certificar_reducido(sig, q, m, phi_ini, simetrico):
    """sig: tamaños enteros; q: densidades enteras (p = q/m). Devuelve cotas rigurosas o None."""
    k = len(sig)
    S = int(sum(sig))
    x_mp = [mp.mpf(int(s)) / S for s in sig]
    P_mp = [[mp.mpf(int(q[a, b])) / m for b in range(k)] for a in range(k)]
    if simetrico:
        orden = np.argsort(np.mod(phi_ini, 2 * np.pi))
        phi = [mp.mpf(0)] * k
        for j, a in enumerate(orden):
            phi[a] = 2 * mp.pi * j / k
        phi = [p_ - phi[0] for p_ in phi]
        F, _ = F_y_J(mp, phi, x_mp, P_mp, k)
        resid = max(abs(f) for f in F)
        if resid > mp.mpf(10) ** (-(DPS - 10)):
            return None
        radio = mp.mpf(0)
    else:
        phi = [mp.mpf(float(p_)) for p_ in phi_ini - phi_ini[0]]
        for _ in range(60):                                   # Newton en alta precisión
            F, J = F_y_J(mp, phi, x_mp, P_mp, k)
            paso = mp.lu_solve(J, mp.matrix(F))
            phi = [phi[0]] + [phi[a] - paso[a - 1] for a in range(1, k)]
            if mp.norm(paso) < mp.mpf(10) ** (-(DPS - 5)):
                break
        if max(abs(float(phi[a]) - float(phi_ini[a] - phi_ini[0])) for a in range(k)) > 0.05:
            return None                                       # Newton se fue lejos: el redondeo rompió el diseño
        radio = mp.mpf(10) ** (-30)
        # test de Krawczyk
        F, J = F_y_J(mp, phi, x_mp, P_mp, k)
        Y = mp.inverse(J)
        caja = [iv.mpf([phi[a] - radio, phi[a] + radio]) for a in range(1, k)]
        phi_iv = [iv.mpf(phi[0])] + caja
        x_iv = [iv.mpf(v) for v in x_mp]
        P_iv = [[iv.mpf(v) for v in fila] for fila in P_mp]
        F_pt, _ = F_y_J(iv, [iv.mpf(p_) for p_ in phi], x_iv, P_iv, k)
        _, J_iv = F_y_J(iv, phi_iv, x_iv, P_iv, k)
        Y_iv = iv.matrix([[iv.mpf(Y[i, j]) for j in range(k - 1)] for i in range(k - 1)])
        I = iv.eye(k - 1)
        Kr = iv.matrix([iv.mpf(phi[a]) for a in range(1, k)]) - Y_iv * iv.matrix(F_pt) + \
             (I - Y_iv * J_iv) * iv.matrix([caja[a] - iv.mpf(phi[a + 1]) for a in range(k - 1)])
        if not all(Kr[a].a > caja[a].a and Kr[a].b < caja[a].b for a in range(k - 1)):
            return None
    # cotas de estabilidad en toda la caja
    phi_caja = [iv.mpf([p_ - radio, p_ + radio]) if a > 0 else iv.mpf(p_) for a, p_ in enumerate(phi)]
    Ms_iv, D_iv = matriz_reducida(iv, phi_caja, [iv.mpf(v) for v in x_mp],
                                  [[iv.mpf(v) for v in fila] for fila in P_mp], k)
    Ms_mid, _ = matriz_reducida(mp, phi, x_mp, P_mp, k)
    E, _ = mp.eigsy(Ms_mid)
    E = sorted(E)
    # ‖Ms(φ) − Ms_mid‖_F ≤ radio_M para todo φ de la caja (cota calculada con intervalos, hacia arriba)
    rad2 = iv.mpf(0)
    for i in range(k):
        for j in range(k):
            rad2 += abs(Ms_iv[i, j] - iv.mpf(Ms_mid[i, j])) ** 2
    radio_M = mp.mpf(iv.sqrt(rad2).b)
    lam2_inf = E[1] - radio_M - mp.mpf(10) ** (-(DPS - 10))     # Weyl + error de eigsy
    D_inf = min(mp.mpf(d.a) for d in D_iv)
    mu = min(sum(Fraction(int(q[a, b]), m) * Fraction(int(sig[b]), S) for b in range(k)) for a in range(k))
    return dict(phi=np.array([float(p_) for p_ in phi]), lam2_inf=float(lam2_inf), D_inf=float(D_inf),
                mu=mu, valido=bool(lam2_inf > 0 and D_inf > 0))


def blowup_hessiano(sig, q, m, phi, semilla):
    """Hessiano completo del blow-up explícito, diagonalizado por bloques de Fourier en Z_m."""
    rng_b = np.random.default_rng(semilla)
    cl = np.repeat(np.arange(len(sig)), sig)               # cluster de cada subcluster
    K = len(cl)
    simb = np.zeros((K, K, m), dtype=complex)
    grados = np.zeros((K, K))
    for u in range(K):
        for v in range(u, K):
            d = int(q[cl[u], cl[v]])
            ind = np.zeros(m)
            if u == v:
                d -= 1                                        # sin lazo: la conexión consigo mismo es el "+1"
                if d <= 0:
                    continue
                S_ = rng_b.choice(np.arange(1, (m - 1) // 2 + 1), d // 2, replace=False)
                ind[S_] = 1; ind[(-S_) % m] = 1
                if d % 2:
                    ind[m // 2] = 1
            else:
                if d == 0:
                    continue
                ind[rng_b.choice(m, d, replace=False)] = 1
            simb[u, v] = np.fft.fft(ind)
            grados[u, v] = ind.sum()
            if v != u:
                simb[v, u] = np.conj(simb[u, v]); grados[v, u] = grados[u, v]
    C = np.cos(phi[cl][:, None] - phi[cl][None, :])
    diag = (C * grados).sum(1)
    minimos, ceros = [], 0
    for w0 in range(0, m, 256):
        H = np.diag(diag)[None] - (C[None] * np.moveaxis(simb[:, :, w0:w0 + 256], 2, 0))
        ev = np.linalg.eigvalsh(H)
        ceros += int((np.abs(ev) < 1e-8 * diag.max()).sum())
        minimos.append(ev.min(1))
    minimos = np.concatenate(minimos)
    N = m * K
    no_trivial = np.sort(np.linalg.eigvalsh(np.diag(diag) - C * simb[:, :, 0].real))[1]
    lam_min = min(no_trivial, minimos[1:].min() if m > 1 else np.inf)
    gmin = (grados.sum(1).min() + 1)
    return dict(N=N, grado_min=int(gmin - 1), mu=gmin / N, lam_min_no_trivial=float(lam_min),
                lam_min_sobre_N=float(lam_min / N), autovalores_nulos=ceros, valido=bool(lam_min > 0 and ceros == 1),
                semilla_saltos=semilla)


def preparar(x, phi, delta, sig, simetrico):
    """Re-resuelve el SDP con tamaños racionales (y, en el caso simétrico, fases exactas y P circulante)."""
    k = len(x)
    xr = sig / sig.sum()
    if not simetrico:
        _, P = SDPClusters(k, delta).resolver(xr, phi)
        return P, phi
    orden = np.argsort(np.mod(phi, 2 * np.pi))
    phi_s = np.zeros(k); phi_s[orden] = 2 * np.pi * np.arange(k) / k
    _, P = SDPClusters(k, delta).resolver(xr, phi_s)
    if P is None:
        return None, phi_s
    pos = np.empty(k, dtype=int); pos[orden] = np.arange(k)
    dist = (pos[None, :] - pos[:, None]) % k
    w = np.array([P[dist == d_].mean() for d_ in range(k)])
    w = (w + w[(-np.arange(k)) % k]) / 2                       # simetría d ↔ −d
    return w[dist], phi_s


def redondear(P, m):
    q = np.floor(P * m + 1e-9).astype(int)                     # hacia abajo: no infla μ
    q = np.minimum(q, q.T)
    np.fill_diagonal(q, m if DIAGONAL_UNO else np.maximum(np.diag(q), 1))
    return q


def certificar(cmp_):
    """Prueba varias racionalizaciones (σ, m). Devuelve la de mayor μ certificado y, aparte, la mejor
    que todavía se puede armar como grafo explícito, con su verificación de Hessiano completo."""
    x, phi, delta = cmp_["x"], cmp_["phi"], cmp_["delta"]
    k = len(x)
    simetrico = es_twisted(x, phi)
    opciones = [np.ones(k, dtype=int)] if simetrico else opciones_tamanos(x)
    salida = dict(metodo="simétrico" if simetrico else "intervalos", valido=False, intentos=[])
    mejor, mejor_bu = None, None
    for sig in opciones:
        P, phi_use = preparar(x, phi, delta, sig, simetrico)
        if P is None:
            salida["intentos"].append(dict(sigma=sig.tolist(), motivo="SDP infactible con tamaños racionales"))
            continue
        for m in M_DENOMINADORES:
            m += m % 2                                           # m par (lo pide el blow-up)
            q = redondear(P, m)
            cert = certificar_reducido(sig, q, m, phi_use, simetrico)
            ok = bool(cert and cert["valido"])
            salida["intentos"].append(dict(sigma=sig.tolist(), m=m, ok=ok, mu=float(cert["mu"]) if ok else None))
            if not ok:
                continue
            fila = dict(sigma=sig, m=m, q=q, mu=float(cert["mu"]), mu_fraccion=str(cert["mu"]),
                        lam2_inf=cert["lam2_inf"], D_inf=cert["D_inf"], phi=cert["phi"])
            if mejor is None or fila["mu"] > mejor["mu"]:
                mejor = fila
            if m <= M_MAX_BLOWUP and sig.sum() ** 2 * m <= BLOWUP_MAX_ENTRADAS and \
                    (mejor_bu is None or fila["mu"] > mejor_bu["mu"]):
                mejor_bu = fila
    if mejor is not None:
        salida.update(valido=True, **{k_: v for k_, v in mejor.items()})
        salida["sigma"] = mejor["sigma"].tolist()
    if mejor_bu is not None:
        salida["blowup"] = blowup_hessiano(mejor_bu["sigma"], mejor_bu["q"], mejor_bu["m"], mejor_bu["phi"],
                                           int(rng.integers(1e9)))
        salida["blowup"].update(sigma=mejor_bu["sigma"].tolist(), m=mejor_bu["m"])
    return salida


certificados = []
for (i, nivel), cmp_ in sorted(comprimidos.items(), key=lambda kv: (kv[0][0], -kv[1]["delta"])):
    t0 = time.time()
    c = certificar(cmp_)
    fila = dict(id=i, delta=cmp_["delta"], k=cmp_["k"], mu_modelo=round(cmp_["mu"], 6), metodo=c["metodo"],
                valido=c["valido"], m=c.get("m"), mu_cert=round(c["mu"], 6) if c["valido"] else None,
                lam2_inf=f"{c['lam2_inf']:.2e}" if c["valido"] else None, D_inf=f"{c['D_inf']:.2e}" if c["valido"] else None,
                seg=round(time.time() - t0, 1))
    if c.get("blowup"):
        b = c["blowup"]
        fila.update(blowup_m=b["m"], blowup_N=b["N"], blowup_mu=round(b["mu"], 6), blowup_valido=b["valido"],
                    blowup_lam_min_N=f"{b['lam_min_sobre_N']:.2e}")
    certificados.append(dict(fila, detalle=c))
    log(f"certificación {i} δ={cmp_['delta']:g}: {'OK' if c['valido'] else 'no'}  " +
        (f"μ={c['mu']:.6f} (σ={c['sigma']}, m={c['m']})" if c["valido"] else f"{len(c['intentos'])} intentos sin éxito"))

tabla_cert = pd.DataFrame([{k_: v for k_, v in c.items() if k_ != "detalle"} for c in certificados])
log("\n" + tabla_cert.to_string(index=False))
guardar(certificados, "certificados.pkl")
tabla_cert.to_csv(os.path.join(RES, "certificados.csv"), index=False)

# %% [markdown]
# ## 9b. Certificación fina (δ ≤ 10⁻⁶)
#
# Con δ chico la certificación de la sección 9 falla: Clarabel cumple el equilibrio F = 0 sólo hasta ~10⁻⁹ y el modo más
# blando del Hessiano reducido es ~δ, así que al re-equilibrar las fases se mueven ~10⁻⁹/δ y se pierde la estabilidad.
# Esta sección certifica los `N_CERT_FINA` mejores sistemas de cada nivel con δ ≤ `DELTA_MAX_FINA`, al estilo de
# O'Keeffe (Lema 1 y Apéndice B):
#
# 1. se fusionan los clusters con la misma fase y se re-resuelve el SDP con margen δ (si el solver deja λ₂ al ras, se
#    prueba con 2δ y 4δ: el template sigue siendo estrictamente estable, con un μ apenas menor);
# 2. con `DPS_FINA` dígitos se corrigen las densidades interiores (0 < p < 1) con la corrección de norma mínima que
#    hace F = 0 exacto (F es lineal en P); la corrección es ~10⁻⁹, mucho menor que el margen;
# 3. tamaños y densidades se redondean a un denominador par D = `D_DEN_FINA` (diagonal (D − 1)/D);
# 4. Newton en alta precisión y test de Krawczyk en una caja de radio `RADIO_FINA`: existe un único equilibrio exacto;
# 5. estabilidad rigurosa sobre toda la caja: D_a > 0 con intervalos y factorización de Cholesky con intervalos de
#    M + v vᵀ (M = X^{-1/2} L X^{-1/2}, v = √x), que prueba L ≻ 0 fuera de las rotaciones;
# 6. μ(T) = min_a Σ_b P_ab x_b exacto (fracción).
#
# Con el Lema 2 de O'Keeffe (blow-up con conjuntos de saltos aleatorios), un template así da grafos simples con grado
# mínimo exactamente μ(T)·n y un equilibrio estrictamente estable no sincronizado, así que μ_c ≥ μ(T).
# Las funciones están definidas en la sección 0b. Resultados en `certificacion_fina.csv` y `certificacion_fina.pkl`;
# el mejor template queda en `template_certificado.json` (mismo formato que `datos/template_reportado.json`).
#

# %%
dps_antes = (mp.dps, iv.dps)
mp.dps = iv.dps = DPS_FINA
cert_fina = []
for delta in [d for d in DELTAS if d <= DELTA_MAX_FINA]:
    en_nivel = [(e["id"], e["familia"], u) for e in estados.values() if not e["id"].startswith("control")
                for u in e["niveles"] if u["delta"] == delta]
    en_nivel.sort(key=lambda t: -t[2]["mu"])
    for ident, familia, u in en_nivel[:N_CERT_FINA]:
        t0 = time.time()
        x, phi, P = fusionar_fases_iguales(np.array(u["x"]), np.array(u["phi"]), np.array(u["P"]))
        fila = dict(id=ident, familia=familia, delta=delta, k=len(x), mu_modelo=round(u["mu"], 7), valido=False)
        for f in FACTORES_MARGEN_FINA:               # si el solver deja λ₂ al ras, se re-resuelve con más margen
            _, P_r = SDPClusters(len(x), f * delta).resolver(x, phi)
            if P_r is None:
                continue
            r = certificar_fino(x, phi, P_r)
            if r["valido"]:
                fila.update(valido=True, margen_sdp=f * delta, mu_cert=float(r["mu"]), pivote_min=r["pivote_min"],
                            c_min=round(r["c_min"], 4))
                cert_fina.append(dict(fila, detalle=r))
                break
        else:
            cert_fina.append(dict(fila, detalle=None))
        log(f"certificación fina {ident} δ={delta:g}: " +
            (f"μ = {fila['mu_cert']:.7f} (margen del SDP {fila['margen_sdp']:g})" if fila["valido"] else "no")
            + f"  ({time.time() - t0:.1f} s)")
mp.dps, iv.dps = dps_antes

tabla_fina = pd.DataFrame([{k_: v for k_, v in c.items() if k_ != "detalle"} for c in cert_fina])
guardar(cert_fina, "certificacion_fina.pkl")
tabla_fina.to_csv(os.path.join(RES, "certificacion_fina.csv"), index=False)
if len(tabla_fina):
    log("\n" + tabla_fina.to_string(index=False))
validos = [c for c in cert_fina if c["valido"]]
if validos:
    mejor_t = max(validos, key=lambda c: c["mu_cert"])
    d = mejor_t["detalle"]
    with open(os.path.join(RES, "template_certificado.json"), "w") as f:
        json.dump(dict(id=mejor_t["id"], delta=mejor_t["delta"], mu=str(d["mu"]), mu_float=float(d["mu"]),
                       D=str(d["D"]), X=[str(v) for v in d["X"]], Q=[[str(v) for v in fila] for fila in d["Q"]],
                       phi=d["phi"]), f, indent=1)
    log(f"mejor template certificado: {mejor_t['id']} (δ = {mejor_t['delta']:g}), μ = {float(d['mu']):.7f} "
        f"→ resultados/template_certificado.json")


# %% [markdown]
# ## 10. Figura final: evolución de los sobrevivientes y de los mejores descartados
#
# Para cada candidato, dos filas:
# - **Arriba**, el sistema en cada etapa: partida (el grafo con sus nodos en θ*, la semilla sintética o el twisted
#   del que salió la patada), entrada a TOPE (después del primer SDP o del SDP de la semilla), un nivel intermedio y
#   la llegada (el último nivel, o el sistema comprimido si sobrevivió). Color y área de cada punto = tamaño x_a.
# - **Abajo**, la matriz P de cada etapa con la franja de tamaños x_a, y a la derecha la curva de μ por todas las
#   etapas: μ(G), SDP inicial, cada nivel de δ, comprimido y certificado. Rojo = O'Keeffe, verde = Canale,
#   punteada = twisted 0.683876, triángulo = techo de TOPE al descartar.
#
# Se dibujan todos los sobrevivientes (controles incluidos) y los `N_TOP_DESCARTADOS` descartados que llegaron más
# lejos: primero por el δ más chico alcanzado y después por μ. Archivos: `evolucion_final_sobrevivientes.png` y
# `evolucion_final_descartados.png`.

# %%
import re

cand_por_id = {c["id"]: c for c in todos}


def etapas(i):
    """Lista de (rótulo, (x, φ, P) o None, μ) para todas las etapas de un candidato."""
    e = estados[i]
    salida = []
    if i in semillas_info:
        s_ = semillas_info[i]
        if s_["tipo"] == "patada":
            padre = cand_por_id.get(s_["padre"])
            if padre is not None and padre.get("A") is not None:
                n = padre["n"]
                salida.append((f"grafo padre {padre['id']}\n({padre['familia']}, n={n})",
                               (np.full(n, 1 / n), padre["theta"], padre["A"] + np.eye(n)), padre["mu_G"]))
            else:
                salida.append((f"twisted padre {s_['padre']}\n({s_['etapa']})", (s_["x_padre"], s_["phi_padre"], None), np.nan))
            salida.append(("patada (SDP crudo)", None, s_.get("mu_crudo", np.nan)))
            salida.append(("patada pre-pulida", (s_["x"], s_["phi"], s_["P0"]), s_["mu0"]))
        elif s_["tipo"] == "pesada":
            salida.append((f"GNN pesada\n(annealing desde {s_['familia_sa']})", (s_["x_gnn"], s_["phi_gnn"], s_["P_gnn"]),
                           s_["mu_gnn"]))
            salida.append(("pesada (SDP crudo)", None, s_.get("mu_crudo", np.nan)))
            salida.append(("pesada pre-pulida", (s_["x"], s_["phi"], s_["P0"]), s_["mu0"]))
        elif s_["tipo"] == "evolucion":
            salida.append((f"padre {s_['padre']}\n({s_['etapa']})", (s_["x_padre"], s_["phi_padre"], s_["P_padre"]), s_["mu_padre"]))
            salida.append(("hijo (SDP crudo)", None, s_.get("mu_crudo", np.nan)))
            salida.append(("hijo pre-pulido", (s_["x"], s_["phi"], s_["P0"]), s_["mu0"]))
        else:
            salida.append((f"sintética {s_.get('geometria', '')} (SDP crudo)", None, s_.get("mu_crudo", np.nan)))
            salida.append(("sintética pre-pulida", (s_["x"], s_["phi"], s_["P0"]), s_["mu0"]))
    else:
        c = cand_por_id[i]; r = c["sdp1"]
        if c.get("A") is not None:
            n = c["n"]
            salida.append((f"grafo {c['familia']}, n={n}\n(nodos en θ*)", (np.full(n, 1 / n), c["theta"], c["A"] + np.eye(n)),
                           c["mu_G"]))
        else:
            salida.append(("semilla de clusters", (c["x0"], c["phi0"], c["P0"]), cantidades(c["x0"], c["phi0"], c["P0"])["mu"]))
        salida.append(("primer SDP", (r["x0"], r["phi0"], r["P1"]) if r["P1"] is not None else None, r["mu1"]))
        if c.get("pulido_sel") is not None:          # pulido de selección
            u_ = c["pulido_sel"]
            salida.append(("pulido de selección", (u_["x"], u_["phi"], u_["P"]), u_["mu"]))
    for u in e["niveles"]:
        salida.append((f"δ={u['delta']:g}", (u["x"], u["phi"], u["P"]) if u["P"] is not None else None, u["mu"]))
    if e["estado"] == "final" and e["niveles"]:
        nl = e["nivel_inicio"] + len(e["niveles"]) - 1
        cmp_ = comprimidos.get((i, len(e["niveles"]) - 1))
        if cmp_ is not None:
            salida.append((f"comprimido (k={cmp_['k']})", (cmp_["x"], cmp_["phi"], cmp_["P"]), cmp_["mu"]))
        cert = [c_ for c_ in certificados if c_["id"] == i and c_["delta"] == e["niveles"][-1]["delta"] and c_["valido"]]
        if cert:
            salida.append(("certificado", None, cert[0]["mu_cert"]))
    return salida


def dibujar_candidato(fig, sub, i):
    et = etapas(i)
    con_sis = [t for t in et if t[1] is not None]
    elegidas = con_sis if len(con_sis) <= 4 else [con_sis[0], con_sis[1], con_sis[len(con_sis) // 2 + 1], con_sis[-1]]
    g = sub.subgridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.3], hspace=0.25, wspace=0.15)
    for j, (rot, (x, phi, P), mu) in enumerate(elegidas):
        ax_c, ax_m = fig.add_subplot(g[0, j]), fig.add_subplot(g[1, j])
        if P is None:            # twisted padre sin P guardada: sólo fases y tamaños
            P = np.eye(len(x))
        tit = f"{rot}\nμ={mu:.5f}" if np.isfinite(mu) else rot
        dibujar_sistema(np.asarray(x), np.asarray(phi), np.asarray(P), tit, ax_c, ax_m, barra_P=False)
        ax_m.set_title("P", fontsize=8)
    for j in range(len(elegidas), 4):
        fig.add_subplot(g[0, j]).axis("off"); fig.add_subplot(g[1, j]).axis("off")
    ax = fig.add_subplot(g[:, 4])
    mus = [m for _, _, m in et]
    xs_ = [j for j, m in enumerate(mus) if np.isfinite(m)]
    ax.plot(xs_, [mus[j] for j in xs_], "o-", ms=4)
    ax.set_xticks(range(len(et)))
    ax.set_xticklabels([r_.split("\n")[0].replace("primer SDP", "SDP1") for r_, _, _ in et], fontsize=6.5, rotation=40, ha="right")
    ax.axhline(M_OBJETIVO, color="tab:red", lw=1); ax.axhline(COTA_CANALE, color="tab:green", lw=1)
    ax.axhline(0.683876, color="0.5", lw=1, ls=":")
    e = estados[i]
    t = re.search(r"techo ([\d.]+)", e.get("motivo") or "")
    if t:
        ax.plot(len(et) - 1, float(t.group(1)), "v", color="tab:orange", ms=7)
    vis = [mus[j] for j in xs_]
    ax.set_ylim(min(vis) - 0.01, max(vis + [M_OBJETIVO]) + 0.006)
    for j in xs_:
        ax.annotate(f"{mus[j]:.4f}", (j, mus[j]), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6); ax.grid(alpha=.3); ax.tick_params(labelsize=7)
    estado = "SOBREVIVIÓ" if e["estado"] == "final" else (e.get("motivo") or e["estado"])
    ax.set_title(f"{i} ({e['familia']})\n{estado}", fontsize=8.5)


def figura_final(ids, titulo, archivo):
    if not ids:
        log(f"{archivo}: no hay candidatos para dibujar"); return
    alto = 6.4 * len(ids)
    fig = plt.figure(figsize=(20, alto))
    G_ = fig.add_gridspec(len(ids), 1, hspace=0.35, top=1 - 0.9 / alto, bottom=0.6 / alto)
    for f_, i in enumerate(ids):
        dibujar_candidato(fig, G_[f_], i)
    fig.suptitle(titulo, fontsize=13, y=1 - 0.3 / alto)
    plt.savefig(os.path.join(RES, archivo), dpi=100, bbox_inches="tight"); plt.show()


def profundidad(i):
    n_ = estados[i]["niveles"]
    return (min(u["delta"] for u in n_) if n_ else 1.0, -(n_[-1]["mu"] if n_ and np.isfinite(n_[-1]["mu"]) else -1))


sobrevivientes = [i for i in finales]
top_desc = sorted([i for i, e in estados.items() if e["estado"] == "descartado" and e["niveles"]],
                  key=profundidad)[:N_TOP_DESCARTADOS]
log(f"Figura final: sobrevivientes {sobrevivientes}; top {N_TOP_DESCARTADOS} descartados {top_desc}")
figura_final(sobrevivientes, "Sobrevivientes: evolución del sistema y de μ por etapa", "evolucion_final_sobrevivientes.png")
figura_final(top_desc, f"Los {N_TOP_DESCARTADOS} descartados que llegaron más lejos", "evolucion_final_descartados.png")

# %% [markdown]
# ## 11. Resumen
#
# Todo queda en `resultados/`:
#
# - `registro.txt`: todo lo impreso, con hora.
# - `dataset_pipeline.pkl`, `gnn_pipeline.pt`, `gnn_verificacion.png`, `gnn_por_familia.csv`: GNN de grafos (paso 1).
# - `candidatos_paso2.pkl`: candidatos con θ*.
# - `hist_sdp1.png`: primer SDP.
# - `seleccion_grafos.csv`, `apartados_seleccion.pkl`: selección de los grafos por pulido.
# - `semillas.pkl`, `hist_semillas.png`, `familias_semillas.csv`, `familias_semillas.png`: semillas sintéticas y patadas.
# - `dataset_pesado.pkl`, `gnn_pesada.pt`, `gnn_pesada_por_familia.csv`, `hallazgos_pesados.pkl`: GNN de sistemas de clusters.
# - `evolucion.pkl`: rondas evolutivas.
# - `estado_tope.pkl`, `descartados_tope.pkl`, `hist_tope.png`, `finales.png`: CMA-ES + SDP con el filtro TOPE.
# - `comprimidos.pkl`, `certificados.pkl`, `certificados.csv`: compresión y certificación.
# - `certificacion_fina.pkl`, `certificacion_fina.csv`, `template_certificado.json`: certificación fina (δ ≤ 10⁻⁶) y el mejor template.
# - `evolucion_final_sobrevivientes.png`, `evolucion_final_descartados.png`: figura final.
#
# **Cómo leer el resultado.** El resultado final es el mayor `mu_cert` válido entre `certificados.csv` y
# `certificacion_fina.csv`: es la conectividad exacta de un template certificado con intervalos, y por lo tanto una cota
# inferior de μ_c. Si en `certificados.csv` además `blowup_valido=True`, hay un grafo explícito (reconstruible con la
# semilla) con ese μ cuyo Hessiano se verificó completo.
#

# %%
filas = []
if len(tabla_cert):
    filas += [dict(id=r.id, delta=r.delta, mu_cert=r.mu_cert, metodo=r.metodo) for r in tabla_cert[tabla_cert.valido].itertuples()]
if len(tabla_fina):
    filas += [dict(id=r.id, delta=r.delta, mu_cert=r.mu_cert, metodo="fina") for r in tabla_fina[tabla_fina.valido].itertuples()]
filas = [f for f in filas if not str(f["id"]).startswith("control")]   # los controles no cuentan como resultado
mejores = pd.DataFrame(filas).sort_values("mu_cert", ascending=False).head(10) if filas else pd.DataFrame()
log("Mejores certificados:\n" + (mejores.to_string(index=False) if len(mejores) else "ninguno"))
if len(mejores):
    m0 = mejores.iloc[0]
    log(f"RESULTADO: μ_c ≥ {m0.mu_cert:.7f}  ({m0.id}, δ = {m0.delta:g}, certificación {m0.metodo})")
log(f"Canale 11/16 = 0.6875; O'Keeffe 0.69291. Fin.")

