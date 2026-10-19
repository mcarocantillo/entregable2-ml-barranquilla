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
# # 8. ¿Supera el modelo nuevo al modelo base?
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# ```{admonition} Respuesta
# :class: important
# **Sí.** El XGBoost geográfico adaptado a clasificación supera al modelo base (logística B) en los dos escenarios,
# con el criterio fijado antes de ver resultados (más de un error estándar de la diferencia pareada en validación
# cruzada) y, en zonas nuevas, también en el test:
#
# - **Zonas nuevas**, con la vecindad física: F1 de test **0.585 frente a 0.493** (+0.092, IC 95 % [+0.034, +0.199]);
#   en validación supera al base en los 5 pliegues (+0.112, SE 0.029).
# - **Zonas conocidas**: +0.19 a +0.23 de F1 en validación, en los 5 pliegues.
#
# **El matiz.** En zonas nuevas la ventaja viene, sobre todo, de la información del entorno: una logística que recibe
# las mismas variables de vecindad empata con el modelo nuevo. En zonas conocidas, en cambio, el modelo nuevo supera
# incluso a la logística con la misma información.
# ```
#
# Corresponde a la diapositiva «¿Supera el modelo nuevo al modelo base?» de la pestaña **6. Resultados**.

# %% tags=["hide-input"]
import sys
from pathlib import Path

RAIZ = Path.cwd().resolve()
while not (RAIZ / "figuras.py").exists():
    RAIZ = RAIZ.parent
sys.path.insert(0, str(RAIZ))

import pandas as pd
import plotly.io as pio

import figuras_proyecto as FP

pio.renderers.default = "notebook"
pd.set_option("display.precision", 3)
AR = FP.arboles_knndm()

# %% [markdown]
# ## 8.1 Cómo se comparó
#
# - **Protocolo del capítulo 5:** pliegues kNNDM (zonas nuevas: 30 grupos, con la geometría del test; zonas conocidas:
#   ocultar manzanas, 6 000 grupos), el mismo test de 7 bloques y diferencias pareadas por pliegue.
# - **Modelos:** el modelo base (logística B v3), el **XGBoost geográfico** y su control (XGBoost, el mismo modelo global
#   sin la parte local). Los hiperparámetros quedan fijos en los elegidos antes (logística: C = 0.01 sin pesos;
#   árboles: los de la ronda 1, profundidad 4 con pesos balanceados, ventana del 15 % y $\alpha$ = 0.75), así que esta
#   ronda no re-busca nada sobre los mismos pliegues.
# - **Variables:** las del modelo base solas y con cada forma de vecindad del capítulo 7 (radios, grafo KNN, y grafo
#   KNN más el estrato de vecinos conocidos).

# %% tags=["hide-input"]
def tabla_modelos(esc):
    m = AR["cv"][esc]["modelos"]
    filas = []
    for k, v in m.items():
        modelo, variables = k.split(" | ", 1)
        filas.append({"Variables": variables, "Modelo": modelo, "F1 CV": v["F1_macro"], "MAE CV": v["MAE"]})
    return pd.DataFrame(filas).pivot_table(index="Variables", columns="Modelo", values="F1 CV", sort=False)


print("F1 macro en validación cruzada kNNDM — zonas nuevas")
tabla_modelos("zonas nuevas")

# %% [markdown]
# ## 8.2 Zonas nuevas

# %% tags=["hide-input"]
FP.fig_nuevo_vs_base("zonas nuevas", "base")

# %% tags=["hide-input"]
FP.tabla_nuevo_vs_base("zonas nuevas")

# %% [markdown]
# - **Con solo las variables del modelo base**, el XGBoost geográfico supera al base en validación (+0.070, SE 0.037,
#   4 de 5 pliegues), pero el test no lo confirma (−0.009, IC [−0.054, +0.150]). Es la misma conclusión de la ronda 1,
#   ahora con la validación correcta.
# - **Con la vecindad física**, lo supera en validación en los **5 pliegues** (+0.112 a +0.132) y en el test: con el
#   grafo KNN, **F1 0.585 frente a 0.493 (+0.092, IC 95 % [+0.034, +0.199])**, y con el grafo KNN más el estrato de
#   los vecinos conocidos, +0.067 (IC [+0.030, +0.183]). Con los radios la diferencia en test es igual de grande
#   (+0.092) pero su intervalo toca el cero.
#
# ## 8.3 Zonas conocidas

# %% tags=["hide-input"]
FP.fig_nuevo_vs_base("zonas conocidas", "base")

# %% tags=["hide-input"]
FP.tabla_nuevo_vs_base("zonas conocidas")

# %% [markdown]
# En zonas con datos cerca (manzanas completas ocultas), el XGBoost geográfico supera al modelo base por **+0.19 a
# +0.23** de F1, en los 5 pliegues y con errores estándar de 0.004 a 0.010. Su mejor versión (con vecindad KNN y el
# estrato de los vecinos) llega a **F1 0.778**.
#
# ## 8.4 ¿Y frente a una logística que recibe la misma información?
#
# El objetivo era superar al modelo base, y se cumple. Pero vale la pena separar cuánto viene del **algoritmo** y
# cuánto de la **información** nueva: se compara el modelo nuevo con una logística que recibe exactamente las mismas
# variables.

# %% tags=["hide-input"]
FP.fig_nuevo_vs_base("zonas nuevas", "misma")

# %% tags=["hide-input"]
FP.fig_nuevo_vs_base("zonas conocidas", "misma")

# %% [markdown]
# - **Zonas nuevas: empate.** Con la misma vecindad, el XGBoost geográfico y la logística quedan a menos de un error
#   estándar (diferencias entre −0.007 y +0.001 en validación). Donde no hay datos cerca, lo que mejora la predicción
#   es **describir el entorno**, y la logística lo aprovecha igual de bien.
# - **Zonas conocidas: gana el modelo nuevo.** Con la misma vecindad física supera a la logística por +0.075 a +0.079, y por
#   +0.022 (SE 0.010, 4 de 5 pliegues) incluso cuando las dos reciben el estrato de los vecinos. Donde hay datos cerca,
#   los modelos locales aprenden el mapa del barrio, que un modelo lineal no puede representar.
#
# ## 8.5 ¿Y frente a su control?

# %% tags=["hide-input"]
tabla_modelos("zonas conocidas")

# %% [markdown]
# Frente al XGBoost global con los mismos hiperparámetros (su control), el XGBoost geográfico suma entre +0.002 y
# +0.007 de F1 en los dos escenarios y con todas las variables: la parte geográfica aporta **poco pero de forma
# consistente**, como ya se había visto en la versión 2 (+0.006 ocultando manzanas, IC [+0.002, +0.015]).
#
# ## 8.6 Conclusión
#
# | Escenario | ¿Supera al modelo base? | ¿Supera a la logística con la misma información? | ¿Supera a su control? |
# |---|---|---|---|
# | Zonas nuevas, variables del base | En CV sí; en test no se confirma | — | Poco |
# | **Zonas nuevas, con vecindad** | **Sí, en CV y en test (+0.09)** | Empata | Poco |
# | **Zonas conocidas** | **Sí, por +0.19 a +0.23** | **Sí, por +0.02 a +0.08** | Poco, pero consistente |
#
# La hipótesis del proyecto se confirma con dos condiciones que salen de los datos: el modelo nuevo necesita
# **información del entorno** para superar al base en zonas sin datos, y su ventaja propia (la de los modelos locales)
# aparece **donde hay datos cerca**, que es el escenario para el que se diseñó. Las limitaciones están en el capítulo 10.
