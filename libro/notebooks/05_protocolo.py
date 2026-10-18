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
# # 5. Cómo se compara: el protocolo de evaluación
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# ```{admonition} Qué responde este capítulo
# :class: tip
# ¿Cómo se decide, sin hacerse trampa, si un modelo es mejor que otro? Corresponde a la pestaña **5. Modelo propuesto
# y protocolo** del dashboard.
# ```

# %% tags=["hide-input"]
import sys
from pathlib import Path

RAIZ = Path.cwd().resolve()
while not (RAIZ / "figuras.py").exists():
    RAIZ = RAIZ.parent
sys.path.insert(0, str(RAIZ))

import pandas as pd
import plotly.io as pio

import figuras as F
import figuras_proyecto as FP

pio.renderers.default = "notebook"
pd.set_option("display.precision", 3)
D, V3 = F.diagnosticos(), FP.v3_e1()

# %% [markdown]
# ## 5.1 Por qué hay que separar zonas
#
# La partición en entrenamiento y prueba se hizo en el Entregable 1 **antes** de mirar los datos: la ciudad se
# dividió en bloques de 2 km y se reservaron 7 bloques (65 145 viviendas) como **test**, que se mira una sola vez.
# Dentro del entrenamiento, la validación cruzada también separa zonas. La razón está en el capítulo 2: con un I de
# Moran de 0.92, viviendas vecinas son casi copias.

# %% tags=["hide-input"]
F.fig_optimismo()

# %% [markdown]
# Con el modelo base, una validación cruzada **aleatoria** (filas mezcladas) da un F1 macro de **0.62**; separando
# zonas, **0.32**: casi el doble. Al mezclar filas, apartamentos de la misma torre y casas de la misma cuadra quedan a
# la vez en entrenamiento y validación, y el modelo «reconoce» la zona. Ese 0.62 describiría qué tan bien se interpola
# dentro de zonas ya vistas, no qué tan bien se generaliza a zonas nuevas.
#
# ## 5.2 Dos escenarios de predicción
#
# ¿Dónde se va a usar el modelo? La respuesta cambia qué es una evaluación honesta, y qué información es legítima:
#
# | | **Zonas nuevas** | **Zonas conocidas** |
# |---|---|---|
# | Pregunta | Estimar el estrato en un barrio sin viviendas estratificadas cerca | Estimar el estrato de una vivienda dentro de un barrio con datos |
# | Uso real | Urbanizaciones nuevas, zonas de expansión | Revisar inconsistencias, completar registros |
# | Cómo se evalúa | Bloques de test de 2 km (y pliegues que imitan esa distancia) | Ocultar manzanas completas dentro del entrenamiento |
# | Distancia al vecino conocido más cercano | ≈ 0.45 km (mediana) | ≈ 0.04 km |
# | ¿Se puede usar el estrato de los vecinos? | Solo el de viviendas fuera de la zona | Sí, el de las viviendas conocidas |
#
# El escenario principal del proyecto es el de **zonas nuevas**; el de zonas conocidas se agregó porque los modelos
# con efectos locales (capítulo 4) están pensados para él.
#
# ## 5.3 El criterio, fijado antes de ver resultados
#
# 1. **Mismo protocolo para todos.** Mismas variables, mismo preprocesamiento (dentro de cada pliegue), mismos
#    pliegues y mismo test.
# 2. **Hiperparámetros solo con la validación cruzada**, con la regla de una desviación estándar: entre las
#    combinaciones a menos de un error estándar de la mejor, la más simple {cite}`hastie2009`.
# 3. **Regla de reemplazo.** Un modelo reemplaza al base solo si su F1 macro en validación cruzada lo supera en más de
#    un error estándar de la **diferencia pareada por pliegue** (los mismos pliegues para ambos).
# 4. **Test una sola vez**, con diferencias pareadas e intervalos de confianza del 95 % por **bootstrap de bloques**
#    (se remuestrean bloques completos, porque las viviendas de un bloque no son independientes).
#
# ## 5.4 Del buffer fijo a la validación kNNDM (versión 3)
#
# En las versiones 1 y 2, la validación cruzada quitaba del entrenamiento las viviendas a menos de **1 km** de cada
# pliegue de validación (un *buffer*). El alcance del correlograma (2.6 km) vaciaba el entrenamiento, así que 1 km fue
# un compromiso. Pero un buffer fijo no dice qué escenario se está midiendo. Siguiendo la sugerencia del profesor de
# usar herramientas de **vecinos más cercanos** {cite}`rubio2026st`, se midió, para cada
# vivienda que se predice, la distancia a la vivienda conocida más cercana:

# %% tags=["hide-input"]
FP.fig_v3_distancias()

# %% tags=["hide-input"]
pd.DataFrame(V3["distancias"]["cuantiles"]).T

# %% [markdown]
# Una zona nueva de 2 km tiene su vecino conocido a **0.45 km** (mediana). Con el buffer, la validación y el test
# evaluaban a **1.4–1.5 km**: un escenario más difícil que el declarado. La **validación kNNDM** (*k-fold nearest
# neighbour distance matching*) {cite}`mila2022,linnenbrink2024` arma los pliegues para reproducir la
# distancia del escenario: agrupa los edificios de entrenamiento con K-Means en $q$ grupos, los reparte en 5 pliegues
# estratificados y elige el $q$ cuya distancia validación → entrenamiento se parece más a la del escenario (distancia
# de Wasserstein entre las dos curvas). Del test solo se usan sus coordenadas.

# %% tags=["hide-input"]
FP.fig_v3_knndm_q()

# %% [markdown]
# El mínimo está en **q = 30** grupos (Wasserstein 0.18 km; la validación con buffer estaba a 1.03 km). En el
# escenario de zonas conocidas, kNNDM elige 6 000 grupos (vecino a 44 m), el equivalente a ocultar manzanas.
#
# ## 5.5 El modelo base con la nueva validación
#
# Con los pliegues kNNDM y sin buffer se repitió la búsqueda de hiperparámetros del modelo base (capítulo 14 del
# Entregable 1, versión 3). Se probaron además dos ideas que el Entregable 1 dejó como próximos pasos: una **logística
# ordinal** y **umbrales de decisión** por estrato.

# %% tags=["hide-input"]
FP.fig_v3_modelos("f1")

# %% tags=["hide-input"]
pd.DataFrame(V3["resumen"]).T

# %% [markdown]
# - **La validación ahora anticipa el test:** la validación cruzada estima 0.46 y el test da 0.49 (con el buffer: 0.32
#   frente a 0.49).
# - **El modelo base de referencia (v3):** logística B con C = 0.01, sin pesos de clase, entrenada con todo el
#   entrenamiento. F1 de test 0.493 (el mismo de antes) con un MAE ordinal menor (0.384 frente a 0.439). A cambio, casi
#   no predice el estrato 5 (recall 0.08).
# - **La ordinal es peor** (F1 de CV −0.03 y MAE más alto): la ubicación no ordena los estratos a lo largo de un solo
#   índice. **Los umbrales no mejoran** en la validación anidada (−0.01); no se adoptan.
#
# **En adelante, «el modelo base» es esta logística B (v3)**, y el protocolo es el de esta sección: pliegues kNNDM,
# regla de un error estándar y test una sola vez. Los resultados de la ronda 1 (capítulo 6) se obtuvieron todavía con
# el buffer y se presentan como tales.
