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
# # 2. Análisis exploratorio
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# Este capítulo corresponde a la **pestaña EDA del dashboard**. Como en el Entregable 1, el EDA se hace **solo con el
# conjunto de entrenamiento** (218 870 viviendas con coordenadas en los bloques de entrenamiento). El dashboard permite
# cambiar a "todas las viviendas" para describir la ciudad completa; aquí se usa entrenamiento salvo que se diga otra
# cosa.

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
train = F.base_eda("train")
print(f"Viviendas de entrenamiento: {len(train):,}".replace(",", " "))

# %% [markdown]
# ## 2.1 Variable objetivo

# %% tags=["hide-input"]
F.fig_distribucion_estrato("train")

# %% [markdown]
# En entrenamiento, los estratos 1 (25.5 %) y 3 (26.0 %) son los más frecuentes, seguidos del 2 (19.9 %) y el 4
# (16.9 %). Los estratos altos son escasos: 6.1 % el 5 y 5.7 % el 6. La razón entre la clase mayor y la menor es de
# **4.6:1**: un desbalance moderado. La clase minoritaria tiene aun así más de 12 000 viviendas, así que no hace falta
# sobremuestrear; basta con una métrica que no premie solo a las clases grandes (F1 macro) y con probar pesos de clase
# como hiperparámetro.

# %% tags=["hide-input"]
F.fig_gradiente_norte_sur("train")

# %% [markdown]
# Al dividir las viviendas en diez franjas de latitud con el mismo número de viviendas, la composición cambia de sur a
# norte: en las dos franjas más al sur casi todo es estrato 1 y 2 (estrato medio 1.5), el estrato medio sube de forma
# casi continua hasta 4.9 en la penúltima franja, donde dominan los estratos 4 a 6, y **baja a 3.3 en la franja más al
# norte**, donde vuelven a aparecer barrios de estratos 1 a 3. El gradiente es fuerte, pero no es una línea recta: hay
# segregación también en sentido este-oeste y por barrios. (En el Entregable 1, con cinco franjas de entrenamiento, el
# estrato medio iba de 1.5 a 4.1.)
#
# **Implicaciones.** La posición norte-sur será, con diferencia, la predictora más fuerte (η² = 0.50 en el
# Entregable 1). Y como viviendas cercanas comparten estrato, una validación que mezcle filas al azar dejaría vecinos
# casi idénticos en entrenamiento y validación: el desempeño medido sería optimista. Esto justifica la validación por
# bloques espaciales.

# %% [markdown]
# ## 2.2 Variables numéricas
#
# El dashboard muestra, para la variable elegida, el histograma (en escala original o log(1 + x)) y las cajas por
# estrato, con el tamaño del efecto de Kruskal-Wallis. Aquí se resumen todas las variables a la vez.

# %% tags=["hide-input"]
F.tabla_resumen_numericas("train").round(2)

# %% [markdown]
# **Forma de las distribuciones.** Las variables de tamaño son muy asimétricas a la derecha: área construida
# (asimetría 4.0), área de terreno (52) y pisos del edificio (10). La mediana es mucho menor que la media y el p99 está
# muy lejos de la mediana. Por eso todas se transforman con log(1 + x) dentro del Pipeline, y antes se recortan
# (winsorizan) los percentiles 0.1 y 99.9 aprendidos en entrenamiento. La **altura** es casi constante (mediana 3 m,
# desviación 0.11): aporta muy poca información.
#
# **Pruebas de normalidad.** Con más de 200 000 filas, cualquier prueba de normalidad rechaza (en el Entregable 1 todas
# lo hicieron con p ≈ 0). Por eso se describen la forma con asimetría, curtosis y percentiles, y las comparaciones entre
# estratos se hacen con pruebas no paramétricas.

# %% tags=["hide-input"]
filas = []
for var, et in F.NUMERICAS.items():
    ef = F.efecto_numerica(var, "train")
    filas.append({"Variable": et, "H de Kruskal-Wallis": round(ef["H"]), "η² (rangos)": ef["eta2_H"],
                  "Fuerza": "fuerte" if ef["eta2_H"] >= 0.14 else "moderada" if ef["eta2_H"] >= 0.06 else "débil"})
tabla_eta = pd.DataFrame(filas).sort_values("η² (rangos)", ascending=False).reset_index(drop=True)
tabla_eta

# %% [markdown]
# Todas las diferencias entre estratos son "significativas" (p < 0.001): con tantos datos, eso no dice nada. Lo que
# importa es el **tamaño del efecto** η², la fracción de la variación (en rangos) que explica el estrato, con los cortes
# habituales de 0.01, 0.06 y 0.14 para efecto pequeño, moderado y grande {cite}`tomczak2014`. La asociación más fuerte
# es la de la **distancia al centro** (η² = 0.32), una variable de ubicación. Las variables físicas
# más asociadas al estrato son el número de **baños**, el **piso de ubicación** (los apartamentos en pisos altos son de
# estratos medios y altos) y el **área construida**. Antigüedad, pisos del edificio y altura tienen efectos débiles.

# %% tags=["hide-input"]
F.fig_numerica("total_banios", log=True)

# %% [markdown]
# **Baños.** Las cajas suben de forma monótona: la mediana es 1 baño en los estratos 1 y 2, 2 en los estratos 3 a 5 y
# 3 en el 6. El traslape entre estratos vecinos es grande (las cajas de 1, 2 y 3 casi coinciden), y ahí es donde un modelo
# basado en estas variables tendrá más dificultades.

# %% tags=["hide-input"]
F.fig_numerica("area_construida", log=True)

# %% [markdown]
# **Área construida.** En escala log la distribución es casi simétrica, lo que confirma que el logaritmo es la
# transformación adecuada. El área crece con el estrato, pero la relación no es lineal: el salto entre los estratos 4 y
# 6 es mucho mayor que entre 1 y 3.

# %% [markdown]
# ## 2.3 Variables categóricas

# %% tags=["hide-input"]
filas = []
for var, et in F.CATEGORICAS.items():
    t = pd.crosstab(train[var].astype(str), train.estrato_num)
    V, chi2 = F.cramer_v(t)
    filas.append({"Variable": et, "Categorías": t.shape[0], "χ²": round(chi2), "V de Cramér": V})
pd.DataFrame(filas).sort_values("V de Cramér", ascending=False).reset_index(drop=True)

# %% tags=["hide-input"]
F.fig_categorica("condicion_predio")

# %% [markdown]
# **Régimen de propiedad** (V = 0.33, asociación moderada). Los predios informales son casi en su totalidad de estratos
# 1 y 2; las unidades en propiedad horizontal (PH) cubren todos los estratos, con más peso en los medios y altos; los
# predios no sometidos a propiedad horizontal (NPH, casas) se concentran en los estratos 1 a 3.

# %% tags=["hide-input"]
F.fig_categorica("uso")

# %% [markdown]
# **Uso** (V = 0.24). Distingue casas de hasta tres pisos de apartamentos en edificios de cuatro pisos o más. Es
# redundante con el régimen: los apartamentos en PH y las unidades PH son en gran parte las mismas viviendas. Las
# categorías con menos del 1 % (vivienda recreacional) se agrupan como "infrecuentes" dentro del one-hot.
#
# **Tipo de vivienda.** La mayoría de las viviendas tienen la categoría "no aplica". La vivienda de interés prioritario
# es casi toda de estrato 1 (97 %); la de interés social se reparte entre los estratos 1 a 4, y la que no es de interés
# social se concentra en los estratos 3 a 6.
#
# Dos categóricas del catastro (`destinacion_economica` y `tipo_planta`) quedaron excluidas en el Entregable 1 por ser
# **degeneradas**: casi todas las viviendas tienen el mismo valor.

# %% [markdown]
# ## 2.4 Correlaciones y valores faltantes

# %% tags=["hide-input"]
F.fig_correlacion("train")

# %% [markdown]
# Se usa la correlación de **Spearman** porque las relaciones son monótonas pero no lineales y las variables tienen
# colas largas. Tres patrones:
#
# - Las variables de **tamaño** están correlacionadas entre sí (área construida con habitaciones ρ = 0.61; habitaciones
#   con baños ρ = 0.52), pero ninguna pareja supera 0.8. En el Entregable 1 los factores de inflación de la varianza
#   (VIF) quedaron por debajo del umbral, así que no se excluyó ninguna variable por multicolinealidad.
# - El **área de terreno** se correlaciona negativamente con el **piso de ubicación** (ρ = −0.69): en propiedad
#   horizontal, cada apartamento registra solo su cuota del lote común, que es más pequeña cuanto más alto y más
#   grande es el edificio.
# - La **distancia al centro** se correlaciona negativamente con el estrato (ρ = −0.43): los estratos altos están al
#   norte, relativamente cerca del centro histórico, y la periferia sur es de estratos bajos.

# %% tags=["hide-input"]
F.fig_faltantes("train")

# %% [markdown]
# Los faltantes son muy pocos: el máximo es 0.12 % en habitaciones. En el Entregable 1 se analizó su **mecanismo**.
# Con cientos de miles de filas, la prueba de Little rechaza MCAR ante cualquier diferencia, así que se combinaron tres
# evidencias: la asociación del faltante con el estrato, la diferencia estandarizada de medias (SMD) entre filas
# completas e incompletas y un clasificador que intenta predecir el faltante. El resultado fue **MAR** para piso de
# ubicación (AUC 0.96), antigüedad (AUC 0.97) y habitaciones (AUC 0.997, por construcción de la regla de coherencia).
# El faltante depende de variables observadas, por ejemplo de una zona concreta de la ciudad, y no del propio valor.
#
# **Decisión.** Imputación por la **mediana** de entrenamiento más un **indicador de faltante**, dentro del Pipeline.
# Con menos del 0.2 % de faltantes, la elección del imputador apenas cambia el modelo; lo importante es aprenderlo solo
# con entrenamiento.

# %% [markdown]
# ## 2.5 Componente espacial

# %% tags=["hide-input"]
F.fig_mapa("estrato_num", "todas")

# %% [markdown]
# El mapa agrega las viviendas en celdas de unos 550 m y colorea cada celda por su estrato medio (aquí con todas las
# viviendas; en el dashboard, con "solo entrenamiento", aparecen huecos rectangulares que son los bloques de test). La
# segregación es evidente: un corredor de estratos altos en el norte, el centro y el oriente de estratos medios, y el
# sur y el suroccidente de estratos 1 y 2. Los cambios de estrato suelen ser bruscos en distancias cortas, de una
# celda a la siguiente: ese es el tipo de patrón local que un modelo lineal con x e y no puede reproducir.

# %% tags=["hide-input"]
F.fig_correlograma()

# %% [markdown]
# **Autocorrelación espacial.** Con una fila por edificio (en una muestra estratificada de entrenamiento) y una matriz
# de pesos de 8 vecinos más cercanos, el **I de Moran global del estrato es 0.913** (p = 0.001 con 999 permutaciones).
# Se eligieron k vecinos y no una banda de distancia porque la densidad de edificios es muy desigual: con una banda
# fija, los edificios de la periferia quedarían sin vecinos. El correlograma muestra cómo se pierde el parecido con la
# distancia: I baja de 0.7 a 250 m hasta cruzar 0.3 hacia los 2.9 km, y se vuelve negativo hacia los 4 km.
#
# **Decisión.** El buffer entre entrenamiento y validación se fijó en **1 km**. El alcance completo (2.9 km) dejaría
# fuera demasiado entrenamiento (capítulo 3 muestra el efecto), así que se aceptó un buffer parcial y se declara como
# limitación: la validación sigue siendo algo optimista.

# %% tags=["hide-input"]
F.fig_tamano_edificios()

# %% [markdown]
# **Dependencia dentro del edificio.** El ICC(1) es 0.991: dentro de un edificio el estrato prácticamente no varía.
# Esto quiere decir que cien apartamentos de una misma torre aportan casi la misma información sobre el estrato que uno
# solo. Es como fotocopiar una hoja 1 000 veces: hay 1 000 hojas, pero una sola página de información. Por eso las
# filas no son datos independientes, y la muestra vale menos de lo que sugiere su número de filas.
#
# *Cómo se mide esa pérdida.* Se usa el efecto de diseño (deff), que dice cuántas veces es menos precisa la muestra que
# una de observaciones independientes. Un deff de 1 significa que no hay repetición; un deff de 100 significa que haría
# falta 100 veces más datos para tener la misma precisión. Se aproxima así:
#
# $$\text{deff} \approx 1 + (\tilde m - 1)\times \text{ICC},$$
#
# donde ICC mide qué tanto se parecen entre sí las viviendas de un mismo edificio (0 = nada, 1 = son idénticas) y
# $\tilde m$ es el tamaño medio del edificio **visto desde una vivienda**.
#
# *Por qué $\tilde m$ no es el promedio por edificio.* Hay dos formas de calcular el "tamaño promedio" de un edificio:
# preguntarle a cada **edificio** cuántas viviendas tiene (la mediana es 1, porque la mayoría son casas), o preguntarle
# a cada **vivienda** cuántas viviendas hay en su edificio y promediar las respuestas. Un ejemplo ilustrativo: en un
# barrio con 99 casas y una torre de 901 apartamentos hay 100 edificios y 1 000 viviendas. Por edificio el promedio es
# 10; pero 901 de las 1 000 viviendas viven en la torre, así que, preguntando a las viviendas, el promedio es
# (99 × 1 + 901 × 901) ÷ 1 000 ≈ 812. Es un promedio ponderado por tamaño: los edificios grandes pesan más porque
# contienen más filas, y las filas son lo que se usa para modelar. Esa segunda forma es la que mide la repetición.
#
# *En este catastro.* La figura lo muestra: el 86 % de los edificios son casas de una sola unidad, pero contienen
# menos de la mitad de las viviendas; los edificios de más de 100 unidades son una fracción mínima de los edificios y
# concentran más de una cuarta parte de las viviendas. Hay torres de hasta 3 045 unidades.

# %% tags=["hide-input"]
v = F.viviendas()
m = v.groupby("edificio_id").size()
m_tilde = float((m ** 2).sum() / m.sum())
icc = 0.991  # ICC(1) calculado en el capítulo 1 del Entregable 1 (ANOVA de una vía por edificio)
deff = 1 + (m_tilde - 1) * icc
print(f"Viviendas: {len(v):,}   Edificios: {len(m):,}   Mediana de unidades por edificio: {int(m.median())}".replace(",", " "))
print(f"Tamaño medio visto desde una vivienda (m~): {m_tilde:.1f}")
print(f"Efecto de diseño ≈ 1 + ({m_tilde:.1f} − 1) × {icc} = {deff:.1f}")
print(f"Tamaño efectivo ≈ {len(v):,} ÷ {deff:.1f} ≈ {len(v) / deff:,.0f}".replace(",", " "))

# %% [markdown]
# **Cómo leer ese 1 870.** No es un conteo de viviendas, de predios, de edificios ni de combinaciones de
# características: es una medida de precisión. Significa que las estimaciones sobre el estrato tienen la precisión de
# unas 1 870 observaciones independientes, no de 332 718. Tampoco significa que sobren o falten filas: hay 168 044
# edificios distintos, muy por encima del mínimo de 20 000. Dos consecuencias prácticas: (1) la partición mantiene
# juntas las unidades de un mismo edificio y de una misma zona, para que una torre nunca quede repartida entre
# entrenamiento y prueba; y (2) los intervalos de confianza remuestrean bloques, no filas (capítulo 3). (El efecto de
# diseño exacto del Entregable 1, 177.8, usa el ICC sin redondear.)

# %% [markdown]
# **Cobertura.** El 16.2 % de las viviendas no tiene coordenadas; el 99.5 % de ellas son predios informales y el
# 96.5 % son de estratos 1 y 2. Es un **sesgo de muestreo espacial**: el modelo solo ve la ciudad formal, y sus
# conclusiones aplican a ella.

# %% [markdown]
# ## 2.6 Lo que no aparece en el dashboard
#
# El EDA del Entregable 1 incluye análisis que no se llevaron al tablero porque no cambian las decisiones del modelo
# base, pero que lo sustentan:
#
# - **Outliers univariados (Tukey, 1.5·IQR).** Marcan entre el 3 % y el 6 % de las filas en áreas, habitaciones y
#   baños, pero son viviendas grandes, no errores; en pisos y altura la regla es inservible porque el IQR es 0. Se
#   decidió **no eliminar filas** por esta regla y tratar las colas con log(1 + x) y winsorización al p99.9.
# - **Outliers multivariados** (Mahalanobis robusta e Isolation Forest): ambos marcan alrededor del 0.6 % de las
#   viviendas, con estratos 5 y 6 sobrerrepresentados. Se conservan: son viviendas reales y grandes.
# - **PCA**: PC1 (29 % de la varianza) es un eje de "tamaño" y PC2 (23 %) un eje de tipología, apartamento moderno en
#   altura frente a casa antigua con lote. Es PC2, no PC1, la más asociada al estrato. La dimensionalidad efectiva es
#   5.4 de 8 variables y los VIF son menores que 3: hay redundancia moderada, insuficiente para reducir dimensiones en
#   el modelo base. K-Means sobre las variables físicas forma grupos que siguen la tipología, no el estrato.
# - **Patrón de puntos**: el índice de Clark-Evans (R = 0.56), la función L de Ripley y DBSCAN con distancia
#   haversine confirman que las viviendas están agrupadas, no distribuidas al azar.
# - **Hotspots**: LISA y Getis-Ord Gi* (con corrección de Benjamini-Hochberg) ubican casi la mitad de los edificios en
#   clusters espaciales significativos y prácticamente ningún outlier espacial.
# - **Escala (MAUP)**: el I de Moran cambia con el tamaño de la celda de agregación; por eso todo el análisis espacial
#   se hizo a nivel de edificio.
# - **Auditoría de fuga**: se excluyeron el número predial (identificador y proxy de manzana) y cualquier rezago
#   espacial del estrato. Ninguna variable individual tiene un AUC univariado sospechoso (el máximo es la posición
#   norte-sur, 0.76).
