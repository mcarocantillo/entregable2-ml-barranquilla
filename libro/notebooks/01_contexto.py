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
# # 1. Contexto del problema
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# Este capítulo corresponde a la **pestaña 1 del dashboard**.

# %% tags=["hide-input"]
# Preparación: el módulo figuras.py (en la carpeta del dashboard) genera las mismas
# figuras que se ven en la app. El renderizador "notebook" incrusta plotly.js en la
# página, así las figuras funcionan sin conexión.
import sys
from pathlib import Path

RAIZ = Path.cwd().resolve()
while not (RAIZ / "figuras.py").exists():
    RAIZ = RAIZ.parent
sys.path.insert(0, str(RAIZ))

import pandas as pd
import plotly.io as pio

import figuras as F

pio.renderers.default = "notebook"
pd.set_option("display.precision", 3)
K = F.kpis()

# %% [markdown]
# ## 1.1 El problema
#
# En Colombia, el **estrato socioeconómico** clasifica los inmuebles residenciales en seis niveles. Su uso principal es
# tarifario: con él se cobran de forma diferenciada los servicios públicos domiciliarios. Los estratos 1, 2 y 3 reciben
# subsidios y los estratos 5 y 6 pagan una contribución adicional, según el régimen de la Ley 142 de 1994. También se
# usa para focalizar programas sociales y para planear la ciudad. Cada alcaldía lo asigna por manzana, con una
# metodología del DANE que considera las características de las viviendas y de su entorno urbano.
#
# El proyecto plantea un problema de **aprendizaje supervisado**: predecir el estrato de cada unidad de vivienda de
# Barranquilla a partir de la información del **catastro**, es decir, sus características físicas y su ubicación.
# Es un problema de **clasificación multiclase** con dos rasgos que condicionan todo el análisis:
#
# - Las clases tienen **orden** (equivocarse de 1 a 2 es menos grave que de 1 a 6). Por eso, además de las métricas
#   de clasificación habituales, se reportan métricas ordinales (MAE ordinal, accuracy ±1 estrato, kappa cuadrático).
# - Los datos tienen **coordenadas**: viviendas cercanas se parecen, y eso obliga a validar con bloques espaciales.
#
# ¿Para qué serviría un modelo así? Para detectar viviendas cuyo estrato asignado no es coherente con sus
# características (posibles errores o desactualizaciones), para apoyar la revisión periódica de la estratificación y
# para estimar el estrato donde no está registrado. No se plantea como reemplazo de la estratificación oficial, que
# incluye información del entorno que el catastro no tiene.

# %% [markdown]
# ## 1.2 Revisión bibliográfica y brecha
#
# Juan Camilo Oñoro realizó una revisión sistemática en Scopus (exportación del 5 de octubre de 2026) con dos ecuaciones
# de búsqueda: una sobre clasificación socioeconómica con aprendizaje supervisado a partir de datos catastrales y
# geoespaciales, y otra sobre modelos de clasificación supervisada mejorados (optimización de hiperparámetros,
# selección de variables, *ensembles*). Se excluyeron el *deep learning*, las redes neuronales y el aprendizaje no
# supervisado.
#
# **Hallazgo principal.** No se encontró ningún estudio que clasifique directamente el estrato o nivel socioeconómico
# de una vivienda usando como predictores sus características catastrales (área, baños, habitaciones, pisos, año de
# construcción) y su ubicación. Los trabajos más cercanos predicen el **precio** de la vivienda (con XGBoost, Random
# Forest o SVR) o usan el nivel socioeconómico como **predictor** de otra cosa, no como variable objetivo
# {cite}`rojas2026,jin2024,ren2024`. El antecedente metodológico más próximo es de 1998 y usa regresión logística con
# datos agregados por sector censal {cite}`sermons1998`.
#
# **Brecha metodológica.** La segunda búsqueda identificó modelos supervisados con evidencia de reducir el error frente
# a clasificadores convencionales, pero que no se han aplicado a este problema. Los más relevantes, en el orden de
# prioridad de la revisión, son:

# %% tags=["hide-input"]
candidatos = pd.DataFrame([
    ["Geographical XGBoost (G-XGBoost)", "Pesos espaciales locales sobre XGBoost; mejora R² y MAE ≈ 17 % frente a XGBoost, GWR y RF geográfico", "El estrato cambia de forma local por barrios (residuos con Moran 0.70)", "grekousis2025"],
    ["GWO-XGBoost / TPE-SVM", "Optimización metaheurística o bayesiana de hiperparámetros", "Espacio de hiperparámetros amplio en XGBoost y SVM", "ojekemi2026,rizkallah2025"],
    ["Ensemble RF + SVM", "RF selecciona variables y SVM clasifica", "Variables catastrales correlacionadas (área, baños, habitaciones)", "zhai2026"],
    ["Bagged TAO Trees", "Árboles oblicuos optimizados; superan a RF, AdaBoost y GB", "Datos tabulares heterogéneos", "carreira2020"],
    ["Rotation Forest", "Rotación PCA antes de cada árbol", "Colinealidad entre variables de tamaño", "sartono2018"],
], columns=["Modelo", "Idea", "Por qué encaja con este problema", "Ref."])
candidatos.drop(columns="Ref.")

# %% [markdown]
# Estos modelos son los candidatos del siguiente entregable {cite}`grekousis2025,carreira2020,sartono2018`. El papel
# de **este** entregable es fijar la **referencia** contra la que se medirán: un modelo base lineal (regresión
# logística), evaluado con un protocolo que respeta la estructura espacial de los datos y comparado contra líneas base
# triviales. Si un modelo novedoso no supera con claridad esta referencia con el mismo protocolo, no aporta.

# %% [markdown]
# ## 1.3 Los datos
#
# | Aspecto | Detalle |
# |---|---|
# | Fuente | Servicio ArcGIS REST de datos abiertos de catastro, Alcaldía de Barranquilla |
# | Fecha de descarga | 15 de septiembre de 2026 |
# | Unidad de observación | Unidad de vivienda (casa o apartamento) del catastro |
# | Tipo de datos | Transversal con componente espacial: una foto del catastro en la fecha de descarga |
# | Variable objetivo | `estrato_num` (1 a 6) |
# | Predictoras | Área construida, área de terreno, habitaciones, baños, pisos del edificio, piso de ubicación, altura, antigüedad, régimen de propiedad, uso, tipo de vivienda y ubicación (x, y, distancia al centro) |
# | Ruta del enunciado | A (clasificación) más la sección de componente espacial; no hay fecha de observación, así que no aplican las secciones temporales |
#
# El año de construcción es un **atributo** de la vivienda, no una marca temporal de la observación: por eso el
# problema es transversal. Las coordenadas son las del centroide del terreno; los apartamentos en propiedad horizontal
# heredan el centroide de su edificio.

# %% [markdown]
# ## 1.4 De los datos descargados a las viviendas analizadas

# %% tags=["hide-input"]
F.fig_embudo()

# %% [markdown]
# La limpieza aplica **reglas que miran una sola fila**, sin aprender nada del conjunto de datos, y por eso se hace
# antes de partir en entrenamiento y prueba:
#
# 1. Se excluyen los usos no habitacionales (comercio, lotes, dotacionales): 36 497 filas.
# 2. Se excluyen las unidades con área construida igual a 0 (11) y las de menos de 10 m² (1 044), que por tamaño son
#    parqueaderos o depósitos registrados como unidades.
# 3. Se excluyen 234 **registros agregados**: filas que describen un edificio completo (más de 20 habitaciones, más de
#    15 baños o más de 2 000 m²) y no una vivienda.
# 4. Se excluyen 12 093 filas sin estrato residencial válido (no aplica u otro). Es la variable objetivo: imputarla
#    sería inventar la respuesta.
#
# Además, los valores imposibles (por ejemplo, el año 2500 o el piso 99, que son códigos de "sin dato") y las
# incoherencias dentro de la fila (más baños que habitaciones + 3, o menos de 6 m² por habitación) se convierten en
# faltantes, con una bandera, sin eliminar la fila. El resultado son **332 718 viviendas**, el 87 % de lo descargado.

# %% tags=["hide-input"]
F.fig_distribucion_estrato("todas")

# %% [markdown]
# En el conjunto completo el estrato 1 es el más frecuente (33 %) y los estratos 5 y 6 suman menos del 10 %. La razón
# entre la clase mayor y la menor es de 8.3:1 en la ciudad completa y de 4.6:1 en el entrenamiento. La diferencia se
# debe en buena parte a que casi todas las viviendas sin coordenadas son predios informales de estratos 1 y 2, que no
# entran al modelo.
#
# **Consecuencia para la evaluación.** Con este desbalance, la accuracy es engañosa: un modelo que solo predijera las
# clases grandes tendría una accuracy aceptable y no serviría para los estratos altos. Por eso la métrica principal es
# el **F1 macro**, que promedia el F1 de las seis clases con el mismo peso.

# %% [markdown]
# ## 1.5 Indicadores de la portada del dashboard

# %% tags=["hide-input"]
pd.DataFrame({
    "Indicador": ["Viviendas analizadas", "Edificios distintos", "Viviendas sin coordenadas", "I de Moran del estrato",
                  "F1 macro del modelo principal en test"],
    "Valor": [f"{K['viviendas']:,}".replace(",", " "), f"{K['edificios']:,}".replace(",", " "),
              f"{K['sin_coord']:.1%}", f"{K['moran']:.3f}", f"{K['f1_test']:.3f}"],
})

# %% [markdown]
# ## 1.6 Flujo de trabajo
#
# El dashboard muestra el flujo en siete pasos. El orden no es decorativo: es lo que evita la **fuga de datos**.
#
# 1. **Descarga** del catastro abierto.
# 2. **Limpieza** con reglas por fila (sección 1.4).
# 3. **Partición espacial**: la ciudad se divide en bloques de 2 km × 2 km y se reserva un 20 % de los bloques para
#    prueba, equilibrando la proporción de estratos. El test queda fijo antes de mirar los datos.
# 4. **EDA solo con entrenamiento**: ninguna decisión del modelo se toma mirando el test.
# 5. **Pipeline**: winsorización, imputación, logaritmos, escalado y one-hot se aprenden dentro de un `Pipeline` de
#    scikit-learn, solo con los datos de entrenamiento de cada fold.
# 6. **Validación cruzada espacial**: cinco folds por bloques y un buffer de 1 km entre entrenamiento y validación.
# 7. **Test una sola vez**, con los hiperparámetros ya fijados.

# %% [markdown]
# ```{admonition} Lo que no aparece en el dashboard
# :class: note
# El detalle completo de la limpieza (umbrales, conteos por regla, diagnóstico de rangos plausibles) y la construcción
# de la base a partir de seis capas del servicio ArcGIS están documentados en los capítulos 1 a 3 del Entregable 1.
# En particular, allí se corrigió un error de uniones muchos-a-muchos que duplicaba cerca del 29 % de las filas.
# ```
