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
# # 6. Versión 3: validación por vecinos más cercanos y vecindad con grafo KNN
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# Este capítulo corresponde a las diapositivas 4.12, 4.13 y 5.12 del dashboard. Recoge la sugerencia del profesor
# de usar herramientas basadas en **k vecinos más cercanos** (KNN) del libro de Modelos Espacio-Temporales
# (Rubio, cap. 2) para revisar el buffer fijo de 1 km. La parte temporal de ese libro no aplica: el catastro es una
# foto de un solo momento.
#
# La partición en entrenamiento y prueba **no cambia** (el test son los mismos 7 bloques de 2 km). Cambian dos
# cosas: cómo se valida dentro del entrenamiento (sección 6.2) y qué variables de entorno recibe el modelo
# (sección 6.4). El detalle del modelo base está en el capítulo 14 del Entregable 1 (versión 3).

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
V3, VK = FP.v3_e1(), FP.vecindad_knn()

# %% [markdown]
# ## 6.1 ¿A qué distancia se predice?
#
# Para cada vivienda que se predice importa a qué distancia está la vivienda más cercana cuyo estrato
# conoce el modelo: con un I de Moran de 0.92, a 50 m el vecino es casi una copia y a 2 km es otro barrio. El
# escenario declarado en el Entregable 1 (una **zona nueva de 2 km cuyo entorno sí está en la base**) tiene ese
# vecino a **0.45 km** (mediana). Con el buffer de 1 km, la validación y el test de la versión 2 evaluaban a
# **1.4–1.5 km**: un escenario más difícil que el declarado. La validación aleatoria, en el otro extremo,
# evalúa a 10 m.

# %% tags=["hide-input"]
FP.fig_v3_distancias()

# %% tags=["hide-input"]
pd.DataFrame(V3["distancias"]["cuantiles"]).T

# %% [markdown]
# ## 6.2 Validación kNNDM
#
# La **validación kNNDM** (*k-fold nearest neighbour distance matching*; Milà et al., 2022; Linnenbrink et
# al., 2024) agrupa los edificios de entrenamiento con K-Means en *q* grupos, los reparte en 5 folds
# estratificados y elige el *q* cuya distancia validación → entrenamiento se parece más a la del escenario
# (distancia de Wasserstein entre las dos curvas). El mínimo está en **q = 30** (0.18 km; la validación de la
# versión 2 estaba a 1.03 km). Así el tamaño de la zona excluida sale de la geometría del problema y no de una
# decisión a mano. Del test solo se usan sus coordenadas.

# %% tags=["hide-input"]
FP.fig_v3_knndm_q()

# %% [markdown]
# ## 6.3 El modelo base con la nueva validación (capítulo 14 del Entregable 1)
#
# Con los folds kNNDM y sin buffer se repitió la búsqueda de hiperparámetros del modelo base y se
# probaron dos ideas que el Entregable 1 dejó como próximos pasos: la **logística ordinal** y **umbrales de
# decisión por estrato** para el F1 macro (aprendidos con predicciones fuera de muestra).

# %% tags=["hide-input"]
FP.fig_v3_modelos("f1")

# %% tags=["hide-input"]
pd.DataFrame(V3["resumen"]).T

# %% [markdown]
# - **La validación ahora anticipa el test**: la CV estima 0.46 y el test da 0.49 (en la versión 2: 0.32 frente
#   a 0.49).
# - **Modelo principal v3**: la misma logística B, con C = 0.01 sin pesos de clase y entrenada con todo el
#   entrenamiento. El F1 de test no cambia (0.493 frente a 0.492), pero el **MAE ordinal baja de 0.439 a 0.384**
#   (diferencia significativa por bootstrap de bloques). A cambio, el estrato 5 casi no se predice (recall 0.08).
# - **La ordinal es peor** (F1 de CV −0.03 y MAE más alto): la ubicación no ordena los estratos a lo largo de un
#   solo índice. **Los umbrales no mejoran** en la validación anidada (−0.01), aunque en test suben 0.02; no se
#   adoptan, porque la decisión se toma en validación.

# %% [markdown]
# ## 6.4 Vecindad con grafo KNN: el lag espacial
#
# El libro del profesor construye un **grafo de k vecinos más cercanos** y usa el **lag espacial**
# $\tilde x = A_{\text{norm}}\,x$ (el promedio de los vecinos) como variable: es una convolución de grafo de un
# salto hecha a mano. Aquí se prueban dos lags, con los hiperparámetros fijos del modelo B v3:
#
# - **Vecindad física KNN**: promedio de seis atributos del catastro (área construida en log, baños,
#   habitaciones, antigüedad, piso y proporción de unidades en propiedad horizontal) en los *k* edificios más
#   cercanos, sin el propio edificio. No usa el estrato, y el catastro tiene esos atributos también para las
#   viviendas que se predicen.
# - **Estrato de vecinos conocidos**: promedio del estrato de los *k* edificios de **entrenamiento** más
#   cercanos. Para que no haya fuga, en las filas de entrenamiento se excluye su propio grupo de kNNDM: así la
#   variable se calcula con vecinos a la misma distancia que al predecir (el principio «solo hasta t − 1» del
#   libro, sec. 2.5.2, llevado al espacio).
#
# **¿Cuántos vecinos?** El libro propone elegir el *k* que maximiza el I de Moran. Aquí el I cae despacio y sin
# máximo interior (0.93 con 2 vecinos, 0.92 con 8, 0.87 con 200), así que ese criterio elegiría el *k* más
# pequeño. Se usó la otra opción del libro: el *k* con mejor F1 en validación, entre 8, 30 y 100.
#
# Se evalúan los **dos escenarios** del capítulo 5, cada uno con sus folds kNNDM: **zonas nuevas** (q = 30, la
# geometría del test) y **zonas conocidas** (ocultar manzanas completas: el vecino conocido está a 44 m de
# mediana; kNNDM elige 6000 grupos y queda a 0.016 km del escenario). Como **control**, se repite la vecindad
# por radios fijos del capítulo 5 (300 m y 1 km) con los mismos folds.

# %% tags=["hide-input"]
FP.fig_moran_k()

# %% tags=["hide-input"]
FP.fig_vecindad_knn("zonas nuevas")

# %% tags=["hide-input"]
FP.fig_vecindad_knn("zonas conocidas")


# %% tags=["hide-input"]
def tabla_cv(esc):
    m = VK["cv"][esc]["modelos"]
    return pd.DataFrame({n: {"F1 macro CV": v["F1_macro"], "Δ F1 vs B": v["dif_F1_vs_B"], "SE Δ": v["SE_dif"],
                             "MAE CV": v["MAE"], "recall E4": v["recall_E4"]} for n, v in m.items()}).T


tabla_cv("zonas nuevas")

# %% tags=["hide-input"]
tabla_cv("zonas conocidas")

# %% tags=["hide-input"]
pd.DataFrame({n: {"F1 macro": v["F1_macro"], "IC 95 % inf": v["IC95_F1"][0], "IC 95 % sup": v["IC95_F1"][1],
                  "MAE": v["MAE"], "kappa": v["kappa"], "recall E4": v["recall"]["E4"]}
              for n, v in VK["test_zonas_nuevas"].items()}).T

# %% [markdown]
# **Zonas nuevas.** En validación, la vecindad física sube el F1 de 0.460 a **0.575** con el grafo KNN
# (k = 8; +0.115, SE 0.015) y a **0.580** con los radios del capítulo 5 (+0.120, SE 0.014): las dos formas de
# describir el entorno son equivalentes. El estrato de los vecinos conocidos suma un poco más (con las dos
# variables, 0.600; +0.140, SE 0.005). En **test**, con el mismo test de siempre:
#
# | Modelo | F1 macro (IC 95 %) | Diferencia con B (IC 95 %) | MAE | Kappa |
# |---|---|---|---|---|
# | Logística B (v3) | 0.493 [0.30, 0.51] | — | 0.384 | 0.85 |
# | + vecindad por radios (cap. 5) | **0.595** [0.43, 0.62] | **+0.102** [+0.022, +0.178] | 0.299 | 0.91 |
# | + vecindad KNN (k = 8) | 0.574 [0.43, 0.60] | +0.081 [+0.015, +0.165] | 0.317 | 0.91 |
# | + vecindad KNN + estrato de vecinos | 0.574 [0.44, 0.59] | +0.081 [+0.033, +0.181] | 0.325 | 0.91 |
#
# Las tres mejoras son significativas (el intervalo excluye el 0) y recuperan sobre todo los estratos altos: el
# recall del estrato 5 pasa de 0.08 a 0.22–0.37 y el del 6 de 0.25 a 0.43–0.45. En test, el estrato de los
# vecinos ya no suma nada sobre la vecindad física.
#
# **Esto corrige la conclusión del capítulo 5.** Con el buffer de 1 km, la vecindad física daba +0.038 en CV con
# un error estándar de 0.045 (no significativa), y en test +0.06 con un intervalo que incluía el 0; el capítulo
# la calificó de «resultado mixto». Con una
# validación que mide el escenario declarado, la misma vecindad da +0.12 en CV y +0.10 en test. El buffer
# escondía su valor: obligaba a predecir a 1.5 km del dato más cercano, donde describir el entorno ayuda menos.
# El control también muestra que la ganancia **no viene del grafo KNN** (los radios rinden igual), sino de
# validar a la distancia correcta.
#
# **Zonas conocidas.** La logística B pasa de 0.551 a 0.693 con la vecindad KNN (k = 30) y a **0.757** al
# agregar el estrato de vecinos (+0.206, SE 0.009). Es prácticamente lo que logró el XGBoost geográfico en el
# capítulo 5 ocultando manzanas (0.754). Los folds no son idénticos, así que la comparación es aproximada, pero
# indica que la ventaja de los árboles en este escenario era, sobre todo, que aprendían el mapa que la logística
# no veía.
#
# ## 6.5 Conclusión de la versión 3
#
# - **Validar en el escenario correcto cambió dos conclusiones.** La validación kNNDM mide la distancia a
#   la que de verdad se predice. Con ella, la vecindad física **sí** mejora las zonas nuevas (F1 de test de 0.49 a
#   0.57–0.60, diferencia significativa), y la validación cruzada anticipa el test.
# - **Modelo recomendado para zonas nuevas:** logística B + vecindad física (radios o grafo KNN, equivalentes).
#   Para zonas conocidas: la misma logística + vecindad + estrato de los vecinos conocidos (F1 ≈ 0.76, a la
#   altura de los árboles).
# - **El grafo KNN del libro del profesor rinde igual que los radios fijos:** lo decisivo fue la validación, no
#   la forma de definir el vecindario. Lo que no cambia es la idea de fondo: el estrato se explica por la zona.
# - **Pendiente (fase 2):** volver a evaluar los árboles (XGBoost o LightGBM) con estas variables y la validación
#   kNNDM. Las cifras de los árboles de los capítulos 4 y 5 se obtuvieron con el buffer de 1 km y no son
#   comparables con las de este capítulo.
