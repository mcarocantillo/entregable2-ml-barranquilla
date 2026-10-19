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
# # 4. El modelo propuesto: XGBoost geográfico adaptado a clasificación
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# ```{admonition} Qué responde este capítulo
# :class: tip
# ¿En qué consiste el modelo nuevo, por qué podría superar al modelo base y con qué otros candidatos de la revisión se
# compara? Corresponde a la pestaña **5. Modelo propuesto y protocolo** del dashboard. Los resultados están en los
# capítulos 6 a 8.
# ```
#
# ## 4.1 La idea original: efectos que cambian de un lugar a otro
#
# Un modelo global, como la logística del capítulo 3 o un XGBoost, aprende **una sola** relación entre las
# variables y el estrato para toda la ciudad. Pero el capítulo 3 mostró que sus errores siguen agrupados por barrios
# (I de Moran de los residuos 0.56): hay zonas enteras donde el modelo se equivoca en la misma dirección. Eso indica que
# la relación cambia de un lugar a otro: por ejemplo, un apartamento de 70 m² con dos baños puede ser estrato 3 en un
# barrio y estrato 5 en otro.
#
# Grekousis {cite}`grekousis2025` propuso el **XGBoost geográfico (G-XGBoost)** para ese tipo de problemas, en
# **regresión**: además de un XGBoost global, entrena modelos **locales** en los que cada observación pesa según su
# distancia a un punto (un kernel espacial, como en la regresión geográficamente ponderada), y combina las predicciones
# del global y de los locales. En sus aplicaciones redujo el error alrededor de un 17 % frente a XGBoost, la regresión
# geográficamente ponderada (GWR) y el Random Forest geográfico.
#
# **Lo nuevo de este proyecto.** No encontramos una versión del G-XGBoost para **clasificación**, ni ninguna aplicación
# al estrato socioeconómico (capítulo 1). La adaptación que sigue es propia.
#
# ## 4.2 La adaptación a clasificación multiclase
#
# 1. **Modelo global.** Un XGBoost multiclase (6 estratos) con todas las viviendas de entrenamiento. Da, para cada
#    vivienda, un vector de probabilidades $p_\text{global}$.
# 2. **Anclas.** En lugar de un modelo local por observación (inviable con cientos de miles de viviendas), se ubica un
#    ancla en el centroide de las viviendas de cada celda de 2 km con al menos 300 viviendas de entrenamiento.
# 3. **Ventana adaptativa.** Para cada ancla $a$, el radio $h_a$ es la distancia a su vecino de entrenamiento número
#    $k$, con $k$ igual a una **fracción** del entrenamiento. En zonas densas la ventana es pequeña y en zonas
#    dispersas, grande; y al ser una fracción, se traslada igual de la validación cruzada (pliegues más pequeños) al
#    modelo final.
# 4. **Kernel bicuadrado.** Cada vivienda $i$ pesa $w_{ia} = \left(1 - (d_{ia}/h_a)^2\right)^2$ si $d_{ia} < h_a$ y 0
#    si no. Con esos pesos se entrena un **XGBoost local** (150 árboles de profundidad 4) en cada ancla. Si en la ventana
#    hay un solo estrato, el modelo local es esa clase con probabilidad 1.
# 5. **Predicción.** Para una vivienda nueva, las probabilidades de los modelos locales que la cubren se promedian con
#    el mismo kernel ($p_\text{local}$); si ninguna ancla la cubre se usa solo el global. La probabilidad final mezcla
#    las dos:
#
# $$p = \alpha\, p_\text{global} + (1 - \alpha)\, p_\text{local}.$$
#
# Los hiperparámetros propios son la **fracción de la ventana** y el **peso del global** $\alpha$: con $\alpha = 1$ es
# un XGBoost global puro, y con $\alpha = 0$ solo cuentan los modelos locales. Por eso el **XGBoost global** con los
# mismos hiperparámetros es el **control natural** del modelo nuevo: la diferencia entre los dos mide cuánto aporta la
# parte geográfica.

# %% tags=["hide-input"]
import pandas as pd

pd.DataFrame([
    ["Modelo global", "XGBoost multiclase, todo el entrenamiento", "Profundidad, número de árboles, pesos de clase"],
    ["Anclas", "Centroides de celdas de 2 km con ≥ 300 viviendas", "Tamaño de celda (fijo)"],
    ["Ventana", "Distancia al vecino k = fracción × n", "Fracción: 5 % o 15 %"],
    ["Kernel", "Bicuadrado", "—"],
    ["Modelos locales", "XGBoost de 150 árboles, profundidad 4, con pesos del kernel", "—"],
    ["Mezcla", "p = α·p_global + (1 − α)·p_local", "α ∈ {0, 0.25, 0.5, 0.75}"],
], columns=["Componente", "Qué es", "Hiperparámetros"])

# %% [markdown]
# **Por qué podría superar al modelo base.** (1) Como todo XGBoost, captura relaciones no lineales e interacciones (un
# baño adicional no pesa igual en una casa que en un apartamento); (2) los modelos locales pueden aprender que la misma
# vivienda corresponde a estratos distintos según el barrio, que es exactamente lo que la metodología del DANE hace al
# calificar primero la zona (sección 1.2).
#
# **Por qué podría no hacerlo.** Los modelos locales aprenden del barrio **si hay datos de ese barrio**. En una zona
# nueva, sin viviendas de entrenamiento cerca, la ventana se llena con barrios vecinos que pueden ser de otro estrato.
# Por eso el capítulo 5 evalúa dos escenarios por separado: predecir zonas nuevas y predecir viviendas dentro de zonas
# con datos.
#
# ## 4.3 Los demás candidatos de la revisión
#
# | Modelo | ¿Original para esta aplicación? | Papel en la comparación | Problema que ataca |
# |---|---|---|---|
# | **XGBoost geográfico (adaptado a clasificación)** | Sí | **Modelo principal** | Relaciones que cambian según la zona |
# | XGBoost | No | Control del anterior: mismo algoritmo sin la parte geográfica | No linealidades e interacciones |
# | Rotation Forest | Sí | Candidato de la revisión {cite}`sartono2018` | Variables de tamaño correlacionadas |
# | SVM RBF (Nyström) + TPE | Sí | Candidato de la revisión {cite}`rizkallah2025` | Fronteras no lineales |
# | RF + SVM en dos etapas | Sí | Candidato de la revisión, agregado en la versión 3 (cap. 9) | Variables correlacionadas |
# | Stacking | Sí | Candidato de la revisión, agregado en la versión 3 (cap. 9) | Modelos que aciertan estratos distintos |
#
# «Original» quiere decir que la revisión no encontró ninguna aplicación del modelo a la clasificación del estrato con
# datos catastrales. El capítulo 9 reúne los nueve candidatos de la revisión y los resultados de todos los que se
# probaron.
#
# **XGBoost.** Árboles de gradiente con 300 árboles y tasa de aprendizaje 0.1; se buscó la profundidad (4, 6 u 8) y si
# usar pesos de clase balanceados.
#
# **Rotation Forest.** Para cada uno de 30 árboles, las variables se reparten al azar en 3 grupos; en cada grupo se
# ajusta un PCA sobre una muestra de un subconjunto aleatorio de clases, y el árbol se entrena con los datos rotados.
# Así los cortes son oblicuos respecto a las variables originales, lo que ayuda con variables correlacionadas.
# Implementación propia.
#
# **SVM RBF (Nyström) + TPE.** Una SVM con kernel RBF exacto sobre 145 000 viviendas es inviable en tiempo. La
# aproximación de Nyström proyecta los datos a unos cientos de dimensiones donde el producto interno aproxima el kernel,
# y sobre esa proyección se entrena una SVM lineal por descenso de gradiente estocástico. Los hiperparámetros se
# buscaron con Optuna (12 ensayos: 4 aleatorios y 8 guiados por TPE).
#
# **Bagged TAO Trees**, también en la revisión, no se implementó: no hay una implementación mantenida en Python y
# programarla desde el artículo era un riesgo desproporcionado. Rotation Forest cubre el lado de "ensemble de árboles
# novedoso".
#
# Todos los candidatos usan las mismas variables del modelo base (físicas + ubicación) y el mismo preprocesamiento,
# ajustado dentro de cada pliegue. En la ronda 2 (capítulo 7) se les agrega información del entorno de la vivienda.
