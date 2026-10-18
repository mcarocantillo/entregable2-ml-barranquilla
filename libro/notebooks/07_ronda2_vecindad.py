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
# # 7. Ronda 2: información del entorno de la vivienda
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# ```{admonition} Qué responde este capítulo
# :class: tip
# La ronda 1 dejó dos pistas: a todos los modelos les falta información de barrio, y el modelo nuevo necesita datos
# cerca. ¿Mejoran los modelos si, además de la vivienda, ven **cómo son las viviendas de alrededor**? Corresponde a la
# pestaña **6. Resultados** del dashboard. El capítulo 8 usa estas variables para responder la pregunta del proyecto.
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

import figuras_proyecto as FP

pio.renderers.default = "notebook"
pd.set_option("display.precision", 3)
VK = FP.vecindad_knn()

# %% [markdown]
# ## 7.1 La vecindad física
#
# Para cada vivienda se resume cómo son las viviendas que la rodean: área construida media (en log), baños, antigüedad,
# proporción de apartamentos o de propiedad horizontal, proporción de predios informales y densidad. Se construye de
# dos formas:
#
# | Forma | Cómo | De dónde viene |
# |---|---|---|
# | **Por radios** | Promedios dentro de discos de **300 m y 1 km** (12 variables) | Primera versión (v2) |
# | **Con grafo KNN** | Promedios en los **k edificios más cercanos** (el *lag espacial* $A_\text{norm}\,x$) | Libro de Modelos Espacio-Temporales del profesor (cap. 2), versión 3 |
#
# Tres reglas evitan la fuga de datos: **solo atributos físicos** (nunca el estrato de los vecinos, que en una zona
# nueva no se conoce; el catastro sí tiene los atributos físicos de todas las viviendas), **sin el propio edificio**
# (repetiría la vivienda) y **sin aprender nada de los datos** (es una regla geométrica; la imputación y el escalado
# siguen dentro del Pipeline).
#
# **El estrato de los vecinos** solo es legítimo si se calcula con viviendas de estrato **conocido**: en zonas
# conocidas, las de alrededor; en zonas nuevas, las de entrenamiento fuera de la zona. Para que se vea igual al
# entrenar que al predecir, en las filas de entrenamiento se calcula excluyendo su propio grupo de kNNDM (el principio
# «solo información disponible al predecir» del libro del profesor, sec. 2.5.2).
#
# **¿Cuántos vecinos?** El libro propone el $k$ que maximiza el I de Moran. Aquí el I cae despacio y sin máximo interior,
# así que se eligió el $k$ con mejor F1 en validación (la otra opción del libro):

# %% tags=["hide-input"]
FP.fig_moran_k()

# %% [markdown]
# ## 7.2 Zonas nuevas con el buffer de 1 km (versión 2): un resultado mixto
#
# Con el protocolo de la ronda 1 (buffer de 1 km), la logística B más la vecindad por radios subió el F1 de **test** de
# 0.492 a **0.554** (+0.062, IC 95 % [−0.027, +0.108], no significativo) y mejoró de forma significativa las métricas
# ordinales (MAE de 0.439 a 0.334). Pero en **validación cruzada** la mejora media (+0.038) no superó un error estándar
# (0.045), tampoco con 15 pliegues (+0.023, SE 0.025), y dependía sobre todo del bloque de test más grande del norte.
# Una variante estricta (vecindad calculada solo dentro de cada partición) dio el mismo F1 (0.553): la mejora no venía
# de usar viviendas de test. **Con el criterio fijado, la vecindad no reemplazaba al modelo base.**
#
# ## 7.3 Zonas nuevas con la validación kNNDM (versión 3): la vecindad sí ayuda
#
# Con la validación que mide el escenario declarado (capítulo 5), la misma pregunta tiene otra respuesta:

# %% tags=["hide-input"]
FP.fig_vecindad_knn("zonas nuevas")


# %% tags=["hide-input"]
def tabla_cv(esc):
    m = VK["cv"][esc]["modelos"]
    return pd.DataFrame({n: {"F1 macro CV": v["F1_macro"], "Δ F1 vs B": v["dif_F1_vs_B"], "SE Δ": v["SE_dif"],
                             "MAE CV": v["MAE"]} for n, v in m.items()}).T


tabla_cv("zonas nuevas")

# %% tags=["hide-input"]
pd.DataFrame({n: {"F1 macro": v["F1_macro"], "IC 95 % inf": v["IC95_F1"][0], "IC 95 % sup": v["IC95_F1"][1],
                  "MAE": v["MAE"], "kappa": v["kappa"], "recall E5": v["recall"]["E5"], "recall E6": v["recall"]["E6"]}
              for n, v in VK["test_zonas_nuevas"].items()}).T

# %% [markdown]
# - **En validación**, la vecindad física sube el F1 de la logística de 0.460 a **0.575** con el grafo KNN (k = 8;
#   +0.115, SE 0.015) y a **0.580** con los radios (+0.120, SE 0.014). Las dos formas son equivalentes.
# - **En test** (los mismos 7 bloques de siempre), el F1 pasa de 0.493 a **0.574** con el grafo KNN (+0.081, IC 95 %
#   [+0.015, +0.165]) y a **0.595** con los radios (+0.102, IC [+0.022, +0.178]): diferencias **significativas**. Se
#   recuperan los estratos altos: el recall del 5 pasa de 0.08 a 0.22–0.37 y el del 6 de 0.25 a 0.43–0.45.
# - El estrato de los vecinos conocidos suma un poco en validación (0.600) y nada en test.
#
# **Por qué cambió la conclusión.** El buffer obligaba a predecir a 1.4–1.5 km del dato más cercano, donde describir el
# entorno ayuda menos; con la distancia real del escenario (0.45 km) la vecindad física aporta de forma clara. La
# ganancia **no viene del grafo KNN** (los radios rinden igual), sino de **validar a la distancia correcta**.
#
# ## 7.4 Zonas conocidas: el entorno ayuda todavía más
#
# Ocultando **manzanas completas** dentro del entrenamiento (la prueba estricta del escenario: en Colombia el estrato
# se asigna por manzana), con pliegues kNNDM:

# %% tags=["hide-input"]
FP.fig_vecindad_knn("zonas conocidas")

# %% tags=["hide-input"]
tabla_cv("zonas conocidas")

# %% [markdown]
# La logística B pasa de 0.551 a 0.693 con la vecindad KNN (k = 30) y a **0.757** con el estrato de los vecinos
# (+0.206, SE 0.009). En la versión 2 (con la partición por manzanas al azar) ya se había visto que, en este escenario,
# los árboles superaban a la logística por +0.20 y que el voto de las 15 viviendas conocidas más cercanas era una regla
# difícil de superar (F1 0.71 ocultando manzanas). El capítulo 8 compara el modelo nuevo con todas estas referencias.
#
# ## 7.5 ¿Dónde ayuda la vecindad? (exploratorio, versión 2)
#
# Con las predicciones de la versión 2, la vecindad ayudó más cuanto más al norte está el bloque (Spearman entre el
# cambio de MAE y la coordenada norte-sur ρ = −0.62, p < 0.001), y su efecto por estrato cambiaba de signo entre la
# validación y el test. Es un análisis posterior a los resultados: sus explicaciones son hipótesis. Una idea que queda
# abierta es medir **qué tan variado es físicamente el entorno** (por ejemplo, la dispersión del área a 300 m), para que
# el modelo sepa cuándo confiar en la vecindad.
#
# ## 7.6 Qué responde la ronda 2
#
# - **Describir el entorno de la vivienda mejora la predicción** en los dos escenarios, también a la logística.
# - En zonas nuevas, la mejora se ve con la validación correcta (kNNDM) y se confirma en el test.
# - Estas variables son las que recibe el modelo nuevo en el capítulo 8.
