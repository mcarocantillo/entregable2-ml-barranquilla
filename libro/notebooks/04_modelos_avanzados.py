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
# Este capítulo corresponde a la **sección 7 de la pestaña 3 del dashboard**. El enunciado del Entregable 2 pide
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
# | **XGBoost geográfico (adaptado a clasificación)** | Propuesta novedosa | Relaciones que cambian según la zona (residuos con Moran 0.70) |
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
# **Resultado de la búsqueda.** En la validación cruzada, ninguna combinación supera al XGBoost global (0.296). En
# general, el F1 mejora al dar más peso al global: con $\alpha = 0$ (solo modelos locales) es 0.277 con la ventana del
# 5 % y 0.271 con la del 15 %. La regla de una desviación estándar eligió la ventana del 15 % y $\alpha$ = 0.75, la
# combinación más cercana al global.
#
# La tabla de cobertura muestra que el problema **no es solo de cobertura**. Con la ventana del 15 %, en promedio el
# 85 % de las viviendas de validación queda dentro de algún modelo local, y aun así los modelos locales solos predicen
# peor que el global. Las viviendas de validación están a más de 1 km de cualquier vivienda de entrenamiento (el
# buffer): los modelos locales se entrenaron con los barrios vecinos, no con el de la vivienda, y como el estrato cambia
# de forma brusca entre barrios, esos vecinos informan peor que un modelo de toda la ciudad. En el test final la
# ventana usa 20 735 viviendas y cubre el 56 % del test.
#
# **Frente a su control,** el XGBoost geográfico mejora el F1 de test en apenas +0.003 (IC [+0.001, +0.006]): una
# diferencia estadísticamente distinguible de cero pero sin importancia práctica. Es lo esperable por diseño: con
# $\alpha$ = 0.75, el 44 % del test no cubierto recibe exactamente la predicción global, y en el 56 % cubierto el global
# pesa el 75 %.
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
# **SVM RBF (Nyström) + TPE** {cite}`rizkallah2025`. Una SVM con kernel RBF exacto sobre 138 000 viviendas es
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
# La búsqueda favoreció pesos de clase balanceados, regularización baja y un $\gamma$ pequeño (≈ 0.006), es decir, un
# kernel suave. Con el entrenamiento completo de cada fold, la SVM elegida obtuvo un F1 de CV de 0.322.

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
# **Selección (solo con la CV).** Ningún modelo cumple el criterio. La SVM RBF es la única con una diferencia media
# positiva frente a la logística (+0.024), pero su error estándar pareado es 0.048: la diferencia depende de un solo
# fold (+0.20 en el fold 4 y cercana a 0 o negativa en los demás). **La logística B sigue siendo el modelo principal.**

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
# **El test confirma la elección.** Ningún modelo supera a la logística B (F1 0.450):
#
# - La **SVM RBF** queda en 0.428 (diferencia −0.022, IC [−0.033, +0.003]) y su MAE ordinal es peor de forma
#   significativa (+0.08, IC [+0.02, +0.12]).
# - **XGBoost** queda en 0.304, muy por debajo de la logística, aunque esa diferencia de F1 no alcanza a ser
#   significativa (−0.146, IC [−0.16, +0.006], p = 0.06). Su MAE ordinal sí es peor de forma significativa (+0.19,
#   IC [+0.04, +0.38]). Llama la atención porque en la CV empataba con la logística (0.296).
# - El **XGBoost geográfico** queda en 0.307, prácticamente igual a su control (sección 4.2).
# - **Rotation Forest** se acerca a la logística en las métricas ordinales (kappa 0.82, accuracy ±1 0.94), pero queda
#   por debajo en F1 (−0.042, IC [−0.058, −0.002]).

# %% [markdown]
# ## 4.5 Por qué los árboles pierden en zonas nuevas

# %% tags=["hide-input"]
F.fig_confusion("XGBoost", normalizar=True)

# %% [markdown]
# El XGBoost casi nunca predice el estrato 5 (3 viviendas en todo el test) y clasifica como 4 el 99 % del estrato 6.
# Los árboles dividen el mapa en rectángulos con cortes en x e y: dentro de las zonas de entrenamiento eso describe bien
# la segregación, pero en una zona nueva el modelo no puede "prolongar" la tendencia y asigna el estrato de la región de
# entrenamiento más parecida. La logística, con un efecto lineal y continuo de la posición, extrapola el gradiente
# sur-norte. Una explicación probable de que XGBoost empate en la CV y pierda en test es que los bloques de test forman
# zonas grandes y continuas, más lejos de los datos de entrenamiento que la mayoría de los bloques de validación; la
# caída de cobertura del XGBoost geográfico entre la CV (85 %) y el test (56 %) apunta en la misma dirección. Sus
# residuos son los más agrupados en el espacio (Moran 0.87).

# %% tags=["hide-input"]
F.fig_importancia_xgb()

# %% [markdown]
# En el XGBoost, las dos variables más importantes son de **tipología** (apartamento en edificio de cuatro pisos o más
# frente a vivienda de hasta tres pisos); luego vienen la posición norte-sur, el tipo de vivienda y el régimen PH. Es
# coherente con el EDA, donde el eje de tipología del PCA era el más asociado al estrato.

# %% [markdown]
# ## 4.6 La SVM cambia los errores de lugar

# %% tags=["hide-input"]
F.fig_confusion("SVM RBF (Nyström) + TPE", normalizar=True)

# %% [markdown]
# La SVM RBF reconoce mejor el **estrato 5** que la logística (recall 0.59 frente a 0.19), pero a costa del 4: el 35 %
# del estrato 4 lo clasifica como 5, y el F1 del 4 baja. Comparte con la logística el problema del **estrato 2**, que
# casi nunca predice (recall 0.01; el 90 % lo clasifica como 3). Con un F1 macro parecido, los dos modelos aciertan
# estratos distintos en la parte alta de la escala, lo que sugiere que combinarlos (*stacking*) podría ayudar con los
# estratos 4 a 6. El estrato 2 parece requerir otra cosa: un modelo ordinal o un ajuste de umbrales.

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
# logística B (0.70): Rotation Forest y la SVM 0.77, XGBoost y su versión geográfica 0.87. La información de barrio
# que le falta al modelo no está en las variables del catastro de la propia vivienda, y los modelos locales no pueden
# aprenderla en zonas sin datos.

# %% [markdown]
# ## 4.8 Conclusión del capítulo
#
# - Con el protocolo de validación espacial, **la regresión logística sigue siendo la mejor referencia**: ningún
#   modelo más complejo la supera ni en la CV (con el criterio fijado de antemano) ni en test, y los modelos de árboles
#   empeoran en zonas nuevas.
# - La adaptación del **XGBoost geográfico a clasificación** funciona, pero no aporta cuando se predice en zonas sin
#   datos de entrenamiento. Es un resultado útil: el beneficio de los efectos locales depende del escenario de
#   predicción. Para evaluarlo en su escenario natural habría que medirlo también **dentro de zonas conocidas** (por
#   ejemplo, ocultando viviendas al azar dentro de los bloques de entrenamiento), y reportar los dos esquemas por
#   separado.
# - La SVM RBF reconoce mejor los estratos altos y la logística los medios: combinarlos es una línea a explorar.
#
# **Limitaciones de este capítulo.** La búsqueda de hiperparámetros fue acotada por el costo de cómputo (rejillas
# pequeñas, 12 ensayos de Optuna en submuestras, 30 árboles en Rotation Forest). La adaptación del XGBoost geográfico
# simplifica el artículo original (anclas en una rejilla y no en cada observación). En Rotation Forest, el subconjunto
# de clases de cada PCA puede incluir las seis clases, una variante menor del algoritmo original. La selección de
# hiperparámetros se hizo con los mismos folds con que se reporta la CV, lo que da cifras de CV algo optimistas para los
# modelos con más búsqueda (en especial la SVM).
