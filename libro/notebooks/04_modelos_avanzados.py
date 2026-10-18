# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 4. Más allá del modelo base: modelos de la revisión bibliográfica
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# Este capítulo corresponde a las diapositivas 5.1 a 5.6 de la pestaña **5. Más allá del modelo base** del dashboard. El enunciado del Entregable 2 pide
# modelos base; este capítulo va un paso más allá y pone a prueba los candidatos de la revisión bibliográfica
# (capítulo 1.2) contra la referencia del capítulo 3, con exactamente el mismo protocolo.

# %% tags=["hide-input"]
import sys
from pathlib import Path

RAIZ = Path.cwd().resolve()
while not (RAIZ / "figuras.py").exists():
    RAIZ = RAIZ.parent
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd
import plotly.io as pio

import figuras as F

pio.renderers.default = "notebook"
pd.set_option("display.precision", 3)
R, D = F.resultados(), F.diagnosticos()
A = R["avanzados"]
B = A["busqueda"]

# %% [markdown]
# ## 4.1 Qué se probó y por qué
#
# | Modelo | Papel en la comparación | Problema que ataca |
# |---|---|---|
# | **XGBoost geográfico (adaptado a clasificación)** | Propuesta novedosa | Relaciones que cambian según la zona (residuos con Moran 0.56) |
# | **XGBoost** | Control del anterior: mismo algoritmo sin la parte geográfica | No linealidades e interacciones |
# | **Rotation Forest** | Ensemble de árboles de la revisión | Variables de tamaño correlacionadas |
# | **SVM RBF (Nyström) + TPE** | SVM no lineal con optimización bayesiana de hiperparámetros | Fronteras no lineales; conecta con el "SVR lineal" del enunciado |
#
# Bagged TAO Trees, también en la revisión, no se implementó: no hay una implementación mantenida en Python y
# programarla desde el artículo era un riesgo desproporcionado para este entregable. Rotation Forest cubre el lado de
# "ensemble de árboles novedoso".
#
# **Protocolo (idéntico al de la logística).** Todos los modelos usan el conjunto de variables B (físicas + ubicación)
# y el mismo preprocesamiento, ajustado dentro de cada fold. Los hiperparámetros se eligen solo con la validación
# cruzada espacial (los mismos 5 folds con buffer de 1 km), con la regla de una desviación estándar cuando hay una
# rejilla. Cada modelo elegido se reentrena con todo el entrenamiento y se evalúa **una sola vez** en test. Las
# diferencias con la logística se miden con bootstrap pareado por bloques.
#
# **Criterio de selección, fijado antes de mirar el test.** La logística B es la referencia. Un modelo nuevo la
# reemplaza solo si su F1 de CV la supera en más de un error estándar de la **diferencia pareada por fold** (los mismos
# 5 folds para ambos; desviación con ddof = 1, dividida por la raíz de 5). Es la regla de una desviación estándar
# aplicada a la comparación entre familias de modelos.

# %% [markdown]
# ## 4.2 La adaptación del XGBoost geográfico
#
# Grekousis {cite}`grekousis2025` propone el XGBoost geográfico para **regresión**: además de un modelo global, entrena
# modelos locales en los que cada observación pesa según su distancia, y combina las predicciones. No encontramos una
# versión para clasificación, ni una aplicación al estrato. La adaptación propuesta aquí es:
#
# 1. **Modelo global.** Un XGBoost multiclase con todas las viviendas de entrenamiento.
# 2. **Anclas.** El centroide de las viviendas de cada celda de 2 km con al menos 300 viviendas de entrenamiento.
#    (El artículo usa un modelo local por observación; las anclas en rejilla son una simplificación por costo.)
# 3. **Ventana adaptativa.** Para cada ancla $a$, el radio $h_a$ es la distancia a su vecino de entrenamiento número
#    $k$, con $k$ igual a una **fracción** del entrenamiento: así la ventana se traslada igual de la CV (folds más
#    pequeños) al modelo final. En zonas densas la ventana es pequeña y en zonas dispersas es grande.
# 4. **Kernel bicuadrado.** Cada vivienda $i$ pesa $w_{ia} = \left(1 - (d_{ia}/h_a)^2\right)^2$ si $d_{ia} < h_a$ y 0 si
#    no. Con esos pesos se entrena un XGBoost local (150 árboles de profundidad 4) por ancla.
# 5. **Predicción.** Para una vivienda nueva, las probabilidades locales se promedian entre las anclas que la cubren,
#    con el mismo kernel; si ninguna la cubre se usa solo el global. La probabilidad final es
#    $p = \alpha\, p_\text{global} + (1-\alpha)\, p_\text{local}$.
#
# Los hiperparámetros propios son la fracción de la ventana (5 % o 15 % del entrenamiento, unas 5 000 o 15 000
# viviendas en un fold) y $\alpha$ (peso del global). $\alpha = 1$ sería el XGBoost global puro, que se evalúa aparte
# como control.

# %% tags=["hide-input"]
F.fig_geoxgb_busqueda()

# %% tags=["hide-input"]
cob = pd.DataFrame({f"ventana {float(k):.0%}": v["cobertura_validacion"] for k, v in B["geoxgb"]["info"].items()},
                   index=[f"fold {i + 1}" for i in range(5)])
cob.loc["media"] = cob.mean()
print("Fracción de viviendas de validación cubiertas por al menos un modelo local:")
cob

# %% [markdown]
# **Resultado de la búsqueda.** En la validación cruzada, ninguna combinación supera al XGBoost global (0.352). El
# F1 tiende a mejorar al dar más peso al global: con $\alpha = 0$ (solo modelos locales) es 0.312 con la ventana del
# 5 % y 0.306 con la del 15 %; la mejor combinación (ventana del 5 %, $\alpha$ = 0.5) llega a 0.324. La regla de una
# desviación estándar eligió la ventana del 15 % y $\alpha$ = 0.75 (F1 de CV 0.320), la combinación más cercana al
# global.
#
# La tabla de cobertura muestra que el problema **no es de cobertura**. Con la ventana del 15 %, en promedio el 86 % de
# las viviendas de validación queda dentro de algún modelo local, y aun así los modelos locales solos predicen peor que
# el global. Las viviendas de validación están a más de 1 km de cualquier vivienda de entrenamiento (el buffer): los
# modelos locales se entrenaron con los barrios vecinos, no con el de la vivienda, y como el estrato cambia de forma
# brusca entre barrios, esos vecinos informan peor que un modelo de toda la ciudad. En el test final la ventana usa
# unas 21 700 viviendas (el 15 % del entrenamiento) y cubre el 94 % del test.
#
# **Frente a su control,** el XGBoost geográfico sube el F1 de test de 0.375 a 0.418 (+0.043, IC [−0.002, +0.058],
# p = 0.10), una diferencia que no alcanza a ser significativa, y baja el MAE ordinal en 0.04 (IC [−0.088, −0.002]).
# En la CV ocurrió lo contrario (0.320 frente a 0.352), así que la ventaja en test no es estable: depende de qué zonas
# caen en la partición.
#
# Es un hallazgo de fondo: los **efectos locales** que el XGBoost geográfico busca capturar se aprenden con datos de la
# misma zona, y la validación espacial pregunta precisamente qué tan bien se predice en **zonas sin datos**. El modelo
# está pensado para interpolar dentro de un territorio observado, no para extrapolar a barrios nuevos.

# %% [markdown]
# ## 4.3 Los otros modelos
#
# **XGBoost.** Árboles de gradiente con 300 árboles y tasa de aprendizaje 0.1. Se buscó la profundidad (4, 6 u 8) y
# si usar pesos de clase balanceados.
#
# **Rotation Forest.** Propuesto por Rodríguez, Kuncheva y Alonso (2006) y evaluado frente a otros clasificadores por
# {cite}`sartono2018`. Para cada uno de 30 árboles, las variables se reparten al azar en 3 grupos; en cada grupo se
# ajusta un PCA sobre una muestra de un subconjunto aleatorio de clases, y el árbol se entrena con los datos rotados.
# Así los cortes del árbol son oblicuos respecto a las variables originales, lo que ayuda con variables
# correlacionadas. Se buscó la profundidad (8 o 14) y los pesos de clase. Implementación propia (`avanzados.py`).
#
# **SVM RBF (Nyström) + TPE** {cite}`rizkallah2025`. Una SVM con kernel RBF exacto sobre 145 000 viviendas es
# inviable en tiempo. La aproximación de Nyström proyecta los datos a unos cientos de dimensiones donde el producto
# interno aproxima el kernel RBF, y sobre esa proyección se entrena una SVM lineal (pérdida hinge) por descenso de
# gradiente estocástico. Los hiperparámetros ($\gamma$ del kernel, número de componentes, regularización y pesos de
# clase) se buscaron con Optuna en 12 ensayos: los 4 primeros aleatorios y los 8 siguientes guiados por el estimador de
# Parzen en árbol (TPE). Por costo, cada ensayo usó una submuestra aleatoria de 30 000 viviendas del entrenamiento de
# cada fold; el elegido se volvió a evaluar con el entrenamiento completo de cada fold. En test, la clase predicha sale
# de ese mismo modelo; un modelo calibrado con el método de Platt aporta solo las probabilidades (AUC, log loss,
# calibración).

# %% tags=["hide-input"]
filas = []
for nombre, clave in [("XGBoost", "xgb"), ("Rotation Forest", "rotf")]:
    for comb, v in B[clave]["grilla"].items():
        filas.append({"Modelo": nombre, "Combinación": comb.replace("|", ", "), "F1 CV": np.mean(v),
                      "Desv.": np.std(v), "Elegido": "✓" if comb == B[clave]["elegido"] else ""})
pd.DataFrame(filas)

# %% tags=["hide-input"]
ens = pd.DataFrame([{**e["params"], "F1 CV (submuestra)": e["f1_submuestra"]} for e in B["svm"]["ensayos"]])
ens.sort_values("F1 CV (submuestra)", ascending=False).head(6).reset_index(drop=True)

# %% [markdown]
# La búsqueda favoreció pesos de clase balanceados, regularización baja ($\alpha$ ≈ 0.0001), 400 componentes de
# Nyström y un $\gamma$ intermedio (≈ 0.13). Con el entrenamiento completo de cada fold, la SVM elegida obtuvo un F1 de
# CV de 0.352, el mismo nivel que el XGBoost.

# %% [markdown]
# ## 4.4 Resultados

# %% tags=["hide-input"]
F.fig_comparacion_avanzados("f1_macro")

# %% tags=["hide-input"]
F.tabla_metricas_avanzados()

# %% tags=["hide-input"]
cvp = pd.DataFrame(A["cv_pareado"]).T[["media", "se", "reemplaza_a_la_logistica"]]
cvp.columns = ["Diferencia media de F1 en CV (vs logística B)", "Error estándar pareado", "¿Reemplaza a la logística?"]
print("Criterio de selección (solo validación cruzada):")
cvp

# %% [markdown]
# **Selección (solo con la CV).** La **SVM RBF cumple el criterio**: supera a la logística en +0.030 de F1 con un
# error estándar pareado de 0.015, y gana en 4 de los 5 folds (en el quinto empata). XGBoost tiene la misma ventaja
# media (+0.030), pero con un error estándar de 0.044 no lo cumple; Rotation Forest y el XGBoost geográfico empatan con
# la logística.
#
# Esa ventaja de la SVM hay que leerla con cautela. Sus cuatro hiperparámetros se buscaron con 12 ensayos de Optuna en
# **los mismos folds** con que se mide la CV, mientras que la logística solo eligió entre una rejilla corta de $C$ y
# pesos de clase: la cifra de CV de la SVM es la más optimista de todas (es la limitación que se declara al final del
# capítulo). El test, que se mira una sola vez, dice si la ventaja se sostiene.

# %% tags=["hide-input"]
F.fig_diferencias()

# %% tags=["hide-input"]
def tabla_comp(clave):
    t = pd.DataFrame(A[clave]).T[["diferencia", "IC95_inf", "IC95_sup", "p_bootstrap"]].astype(float)
    t.columns = ["Diferencia", "IC 95 % inf.", "IC 95 % sup.", "p (bootstrap)"]
    return t


print("Diferencia de F1 macro (positivo = mejor que la referencia):")
display(tabla_comp("comparaciones_f1"))
print("Diferencia de MAE ordinal (negativo = mejor que la referencia):")
display(tabla_comp("comparaciones_mae"))

# %% [markdown]
# **El test no confirma la ventaja de la SVM.** Ningún modelo supera a la logística B (F1 0.492):
#
# - La **SVM RBF** queda en 0.453 (diferencia −0.039, IC [−0.079, +0.012], no significativa) y su MAE ordinal es
#   peor de forma significativa (+0.08, IC [+0.02, +0.14]). También es peor como clasificador probabilístico: AUC 0.82
#   frente a 0.90 y log loss 1.17 frente a 0.91.
# - **XGBoost** queda en 0.375 (−0.117, IC [−0.172, +0.030], p = 0.12), lejos de la logística aunque la diferencia no
#   alcance a ser significativa. Llama la atención porque en la CV estaba arriba (0.352).
# - El **XGBoost geográfico** queda en 0.418, algo por encima de su control (sección 4.2).
# - **Rotation Forest** queda por debajo en F1 (0.450; −0.042, IC [−0.071, +0.019]), pero es el **mejor en las
#   métricas ordinales**: su MAE es menor que el de la logística de forma significativa (−0.05, IC [−0.070, −0.019]),
#   con kappa 0.85 y accuracy ±1 0.96. Se equivoca de clase con una frecuencia parecida, pero por menos estratos.
#
# **Decisión.** Con el criterio fijado de antemano, la SVM reemplazaría a la logística. **Mantenemos la logística B
# como modelo principal** y lo declaramos como una desviación del criterio: (1) la ventaja de la SVM en CV es la más
# expuesta al sobreajuste de la búsqueda; (2) en test no se sostiene, y sus errores ordinales y sus probabilidades
# son peores, que es lo que se usa en el simulador y en el mapa de incertidumbre del dashboard; y (3) la logística es
# interpretable. Una CV anidada (la búsqueda de la SVM repetida dentro de cada fold) resolvería el empate sin mirar el
# test.

# %% [markdown]
# ## 4.5 Por qué los árboles pierden en zonas nuevas

# %% tags=["hide-input"]
F.fig_confusion("XGBoost", normalizar=True)

# %% [markdown]
# El XGBoost colapsa la escala en dos bloques: casi no predice el estrato 3 (acierta solo el 5 % de las viviendas de
# estrato 3; el 76 % va al 2) y manda al estrato 4 el 80 % del estrato 5 y el 86 % del 6. Los árboles dividen el mapa
# en rectángulos con cortes en x e y: dentro de las zonas de entrenamiento eso describe bien la segregación, pero en
# una zona nueva el modelo no puede "prolongar" la tendencia y asigna el estrato de la región de entrenamiento más
# parecida. La logística, con un efecto lineal y continuo de la posición, extrapola el gradiente sur-norte. Una
# explicación probable de que XGBoost gane en la CV y pierda en test es que los bloques de test forman zonas grandes y
# continuas, más lejos de los datos de entrenamiento que la mayoría de los bloques de validación. Sus residuos están
# más agrupados en el espacio que los de la logística (Moran 0.72 frente a 0.56).

# %% tags=["hide-input"]
F.fig_importancia_xgb()

# %% [markdown]
# En el XGBoost, las dos variables más importantes son de **tipología** (vivienda de hasta tres pisos y unidad predial
# en PH); luego vienen la posición norte-sur y este-oeste y el uso de apartamento en edificio de cuatro pisos o más. Es
# coherente con el EDA, donde el eje de tipología del PCA era el más asociado al estrato.

# %% [markdown]
# ## 4.6 La SVM cambia los errores de lugar

# %% tags=["hide-input"]
F.fig_confusion("SVM RBF (Nyström) + TPE", normalizar=True)

# %% [markdown]
# La SVM RBF reconoce mejor los extremos que la logística: el **estrato 5** (recall 0.58 frente a 0.42), el 6 (0.27
# frente a 0.21) y el 1 (0.87 frente a 0.79). Lo paga en el **estrato 2** (recall 0.30 frente a 0.60: el 38 % lo
# manda al 1 y el 30 % al 3) y en el 4, del que clasifica como 5 el 40 %. Con un F1 macro parecido, los dos modelos
# aciertan estratos distintos, lo que sugiere que combinarlos (*stacking*) podría ayudar, sobre todo con los estratos
# 4 a 6. El estrato 4 sigue siendo el más difícil para ambos (recall 0.28 en los dos).

# %% [markdown]
# ## 4.7 Residuos espaciales

# %% tags=["hide-input"]
mr = pd.DataFrame(D["moran_residuos"]).T[["I", "residuo_medio"]]
mr.index = [{"A_fisicas": "Logística A", "B_fisicas+ubicacion": "Logística B"}.get(i, i) for i in mr.index]
mr.columns = ["I de Moran de los residuos", "Residuo medio"]
mr

# %% tags=["hide-input"]
F.fig_mapa_residuos_modelo("XGBoost geográfico")

# %% [markdown]
# Ninguno de los modelos elimina la autocorrelación de los residuos, y todos los modelos nuevos la aumentan frente a la
# logística B (0.56): Rotation Forest 0.66, el XGBoost geográfico 0.69, XGBoost 0.72 y la SVM 0.76. La información de
# barrio que le falta al modelo no está en las variables del catastro de la propia vivienda, y los modelos locales no
# pueden aprenderla en zonas sin datos.

# %% [markdown]
# ## 4.8 Conclusión del capítulo
#
# - Con el protocolo de validación espacial, **la regresión logística sigue siendo la mejor referencia en test**:
#   ningún modelo más complejo la supera. La SVM RBF pasa el criterio de CV, pero su ventaja no se sostiene en test y
#   sus errores son más grandes; mantener la logística es una desviación declarada del criterio.
# - **Rotation Forest** es la alternativa más interesante si importa el error ordinal: F1 parecido y errores más
#   cortos.
# - La adaptación del **XGBoost geográfico a clasificación** funciona, pero no aporta de forma estable cuando se
#   predice en zonas sin datos de entrenamiento. El beneficio de los efectos locales depende del escenario de
#   predicción; el capítulo 5 lo mide también **dentro de zonas conocidas**.
# - La SVM RBF reconoce mejor los extremos de la escala y la logística el estrato 2: combinarlos es una línea a
#   explorar.
#
# **Limitaciones de este capítulo.** La búsqueda de hiperparámetros fue acotada por el costo de cómputo (rejillas
# pequeñas, 12 ensayos de Optuna en submuestras, 30 árboles en Rotation Forest). La adaptación del XGBoost geográfico
# simplifica el artículo original (anclas en una rejilla y no en cada observación). En Rotation Forest, el subconjunto
# de clases de cada PCA puede incluir las seis clases, una variante menor del algoritmo original. La selección de
# hiperparámetros se hizo con los mismos folds con que se reporta la CV, lo que da cifras de CV algo optimistas para los
# modelos con más búsqueda (en especial la SVM).
