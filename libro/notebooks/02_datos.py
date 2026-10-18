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
# # 2. Datos y análisis exploratorio
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# ```{admonition} Qué responde este capítulo
# :class: tip
# ¿Con qué datos se trabaja y qué rasgos de los datos condicionan los modelos y su evaluación? Corresponde a las
# pestañas **2. Datos y calidad** y **3. Análisis exploratorio** del dashboard. El detalle completo está en el libro del
# Entregable 1.
# ```

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
K = F.kpis()
train = F.base_eda("train")

# %% [markdown]
# ## 2.1 La fuente
#
# | Aspecto | Detalle |
# |---|---|
# | Fuente | Servicio ArcGIS REST de datos abiertos de catastro, Alcaldía de Barranquilla |
# | Fecha de descarga | 15 de septiembre de 2026 |
# | Unidad de observación | Unidad de vivienda (casa o apartamento) del catastro |
# | Tipo de datos | Transversal con componente espacial: una foto del catastro en la fecha de descarga |
# | Variable objetivo | `estrato_num` (1 a 6) |
# | Predictoras | Área construida, área de terreno, habitaciones, baños, pisos, piso de ubicación, altura, antigüedad, régimen de propiedad, uso, tipo de vivienda y ubicación (x, y, distancia al centro) |
#
# El año de construcción es un **atributo** de la vivienda, no una marca temporal de la observación: el problema es
# transversal. Las coordenadas son las del centroide del terreno; los apartamentos heredan el centroide de su edificio,
# y las viviendas informales sin terreno propio, el centroide de su manzana (sección 2.2).
#
# ## 2.2 De los datos descargados a las viviendas analizadas

# %% tags=["hide-input"]
F.fig_embudo()

# %% [markdown]
# La limpieza aplica **reglas que miran una sola fila**, sin aprender nada del conjunto de datos, y por eso se hace
# antes de partir en entrenamiento y prueba: se excluyen los usos no habitacionales (36 497 filas), las unidades de
# menos de 10 m² (1 055), los registros que describen un edificio completo (234) y las filas sin estrato residencial
# válido (12 093). Los valores imposibles (por ejemplo, el año 2500 o el piso 99, códigos de "sin dato") y las
# incoherencias dentro de la fila se convierten en faltantes, con una bandera. Quedan **332 718 viviendas**, el 87 %
# de lo descargado.
#
# El 16.2 % de esas viviendas (53 894, casi todas mejoras informales de estratos 1 y 2) no tiene coordenadas propias.
# A 43 594 de ellas se les asigna el **centroide de su manzana** (solo en manzanas compactas; error mediano de 31 m).
# Así, el **96.9 %** de las viviendas entra al modelo; quedan fuera 10 300 (3.1 %), casi todas de estrato 1.

# %% tags=["hide-input"]
pd.DataFrame({
    "Indicador": ["Viviendas analizadas", "Predios matriz (edificios, conjuntos o lotes)", "Viviendas sin coordenadas",
                  "I de Moran del estrato", "F1 macro del modelo base en test"],
    "Valor": [f"{K['viviendas']:,}".replace(",", " "), f"{K['edificios']:,}".replace(",", " "),
              f"{K['sin_coord']:.1%}", f"{K['moran']:.3f}", f"{K['f1_test']:.3f}"],
})

# %% [markdown]
# ## 2.3 La variable objetivo: desbalanceada y ordinal

# %% tags=["hide-input"]
F.fig_distribucion_estrato("train")

# %% [markdown]
# El EDA se hace **solo con el conjunto de entrenamiento** (257 273 viviendas). El estrato 1 es el más frecuente
# (30.0 %), seguido del 3 (22.9 %), el 2 (22.7 %) y el 4 (14.4 %); los estratos altos son escasos (5.2 % el 5 y 4.8 % el
# 6). La razón entre la clase mayor y la menor es **6.2:1**. Con este desbalance la accuracy engaña, así que la
# **métrica principal es el F1 macro**, que promedia el F1 de los seis estratos con el mismo peso.

# %% tags=["hide-input"]
F.fig_gradiente_norte_sur("train")

# %% [markdown]
# En diez franjas de latitud con el mismo número de viviendas, el estrato medio sube de 1.5 en el sur a 4.8 en la
# penúltima franja y baja a 3.0 en la franja más al norte. La posición norte-sur será la predictora más fuerte
# (η² = 0.46), y la segregación obliga a validar separando zonas (capítulo 5).
#
# ## 2.4 Variables numéricas y categóricas

# %% tags=["hide-input"]
F.tabla_resumen_numericas("train").round(2)

# %% tags=["hide-input"]
filas = []
for var, et in F.NUMERICAS.items():
    ef = F.efecto_numerica(var, "train")
    filas.append({"Variable": et, "η² (rangos)": ef["eta2_H"],
                  "Fuerza": "fuerte" if ef["eta2_H"] >= 0.14 else "moderada" if ef["eta2_H"] >= 0.06 else "débil"})
pd.DataFrame(filas).sort_values("η² (rangos)", ascending=False).reset_index(drop=True)

# %% [markdown]
# **Forma.** Las variables de tamaño son muy asimétricas a la derecha (área construida 4.0, área de terreno 56), así que
# se transforman con log(1 + x) dentro del Pipeline, tras recortar los percentiles 0.1 y 99.9. La **altura** es casi
# constante y aporta muy poco.
#
# **Asociación con el estrato.** Con tantos datos todas las diferencias son "significativas"; importa el **tamaño del
# efecto** η² (0.01, 0.06 y 0.14 para pequeño, moderado y grande {cite}`tomczak2014`). Las más asociadas son los
# **baños** (0.33), la **distancia al centro** (0.28), el **piso de ubicación** (0.25) y el **área construida** (0.17).

# %% tags=["hide-input"]
F.fig_numerica("total_banios", log=True)

# %% tags=["hide-input"]
filas = []
for var, et in F.CATEGORICAS.items():
    t = pd.crosstab(train[var].astype(str), train.estrato_num)
    V, chi2 = F.cramer_v(t)
    filas.append({"Variable": et, "Categorías": t.shape[0], "V de Cramér": V})
pd.DataFrame(filas).sort_values("V de Cramér", ascending=False).reset_index(drop=True)

# %% tags=["hide-input"]
F.fig_categorica("condicion_predio")

# %% [markdown]
# **Régimen de propiedad** (V = 0.33, asociación moderada): los predios informales son casi todos de estratos 1 y 2
# (98 %); las unidades en propiedad horizontal cubren todos los estratos, con más peso en los medios y altos; las casas
# (NPH) se concentran en los estratos 1 a 3. El **uso** (V = 0.26) es en parte redundante con el régimen.
#
# ## 2.5 Correlaciones y faltantes

# %% tags=["hide-input"]
F.fig_correlacion("train")

# %% [markdown]
# Se usa Spearman porque las relaciones son monótonas pero no lineales. Las variables de tamaño están correlacionadas
# (área con habitaciones ρ = 0.61), pero ninguna pareja supera 0.8 y los VIF quedaron por debajo del umbral: no se
# excluyó ninguna variable. Los **faltantes** son muy pocos (máximo 0.12 %) y de mecanismo MAR: se imputan con la
# mediana de entrenamiento más un indicador, dentro del Pipeline.

# %% [markdown]
# ## 2.6 El componente espacial: el estrato es una variable de zona

# %% tags=["hide-input"]
F.fig_mapa("estrato_num", "todas")

# %% [markdown]
# Un corredor de estratos altos en el norte, el centro y el oriente de estratos medios, y el sur y el suroccidente de
# estratos 1 y 2. Los cambios suelen ser bruscos de una celda a la siguiente: un patrón local que un modelo lineal con
# x e y no puede reproducir, y que motiva el modelo propuesto (capítulo 4).

# %% tags=["hide-input"]
F.fig_correlograma()

# %% [markdown]
# **Autocorrelación espacial.** Con una fila por edificio y 8 vecinos más cercanos, el **I de Moran del estrato es
# 0.918** (p = 0.001). El correlograma baja de 0.74 a menos de 250 m hasta cruzar 0.3 hacia los 2.6 km. Esta
# dependencia es la razón por la que la validación debe separar zonas (capítulo 5).

# %% tags=["hide-input"]
F.fig_tamano_edificios()

# %% tags=["hide-input"]
v = F.viviendas()
m = v.groupby(F.grupos_predio(v)).size()      # grupo = predio matriz; mejoras aparte
m_tilde = float((m ** 2).sum() / m.sum())
icc = 0.992
deff = 1 + (m_tilde - 1) * icc
print(f"Tamaño medio visto desde una vivienda (m~): {m_tilde:.1f}")
print(f"Efecto de diseño ≈ 1 + ({m_tilde:.1f} − 1) × {icc} = {deff:.1f}")
print(f"Tamaño efectivo ≈ {len(v):,} ÷ {deff:.1f} ≈ {len(v) / deff:,.0f}".replace(",", " "))

# %% [markdown]
# **Dependencia dentro del predio matriz.** El ICC(1) es 0.992: dentro de un edificio o conjunto el estrato casi no
# varía. El efecto de diseño de Kish, $\text{deff} \approx 1 + (\tilde m - 1)\,\text{ICC}$ con $\tilde m$ el tamaño de
# grupo visto desde una vivienda, es ≈ 144: las 332 718 viviendas tienen la precisión de unas **2 300 observaciones
# independientes**. Dos consecuencias: la partición mantiene juntas las unidades de un mismo edificio y de una misma
# zona, y los intervalos de confianza remuestrean bloques, no filas.
#
# ## 2.7 Qué decisiones salen de este capítulo
#
# | Hallazgo | Decisión |
# |---|---|
# | Desbalance 6.2:1 | Métrica principal F1 macro; pesos de clase como hiperparámetro |
# | Estrato ordinal | Métricas ordinales (MAE, accuracy ±1, kappa cuadrático) |
# | Colas largas | log(1 + x) y winsorización dentro del Pipeline |
# | Faltantes < 0.2 %, MAR | Mediana + indicador, dentro del Pipeline |
# | Moran 0.92, cambios bruscos entre barrios | Validación separando zonas (cap. 5); un modelo con efectos locales (cap. 4) |
# | ICC 0.99, tamaño efectivo ≈ 2 300 | Partición por edificios y bloques; bootstrap por bloques |
#
# El EDA del Entregable 1 incluye además outliers univariados y multivariados (se conservan: son viviendas reales),
# PCA (PC1 es un eje de tamaño y PC2 de tipología), patrón de puntos, LISA y Gi*, el efecto de la escala (MAUP) y una
# auditoría de fuga (ninguna variable tiene un AUC univariado sospechoso; el máximo es la posición norte-sur, 0.75). El
# dashboard los muestra en la pestaña 3.
