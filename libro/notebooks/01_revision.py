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
# # 1. El problema y la revisión bibliográfica
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# ```{admonition} Qué responde este capítulo
# :class: tip
# ¿Qué problema se quiere resolver, qué dice la literatura sobre él y qué modelo, **no usado antes en esta
# aplicación**, se propone para superar al modelo base? Corresponde a la pestaña **1. Problema y revisión** del
# dashboard.
# ```

# %% tags=["hide-input"]
import pandas as pd

pd.set_option("display.max_colwidth", 120)

# %% [markdown]
# ## 1.1 El problema
#
# En Colombia, el **estrato socioeconómico** clasifica los **inmuebles residenciales** (no los hogares) en seis
# niveles. Su uso principal es tarifario: con él se cobran de forma diferenciada los servicios públicos domiciliarios
# con subsidios cruzados (Ley 142 de 1994). Los estratos 1, 2 y 3 reciben subsidios, pagados con fondos de solidaridad
# (artículo 89); el estrato 4 paga el costo del servicio; y los estratos 5 y 6, junto con los usuarios comerciales e
# industriales, pagan una contribución. Los topes de subsidio dependen del servicio: hasta 50 %, 40 % y 15 % del consumo
# básico para los estratos 1, 2 y 3 en energía y gas (artículo 99.6), y 70 %, 40 % y 15 % en acueducto, alcantarillado y
# aseo (Ley 1450 de 2011). Un estrato mal asignado hace que el subsidio llegue a quien no lo necesita o no llegue a
# quien sí. También se usa para focalizar programas sociales y para planear la ciudad. Cada alcaldía lo asigna con una
# metodología del DANE que considera las características de las viviendas y de su entorno urbano (sección 1.2).
#
# El estrato es un instrumento colombiano: según el DANE, el cobro diferencial por estratos «se institucionalizó sin
# que existiera un modelo similar en otros países» {cite}`dane2015conceptual`. Esto importa para la revisión
# bibliográfica (sección 1.3): la variable que se quiere predecir no existe fuera de Colombia.
#
# El proyecto plantea un problema de **aprendizaje supervisado**: predecir el estrato de cada unidad de vivienda de
# Barranquilla a partir de la información del **catastro abierto**, es decir, sus características físicas (área, baños,
# habitaciones, pisos, antigüedad, régimen de propiedad) y su ubicación. Es un problema de **clasificación multiclase**
# con dos rasgos que condicionan todo el análisis:
#
# - Las clases tienen **orden** (equivocarse de 1 a 2 es menos grave que de 1 a 6). Además de las métricas de
#   clasificación habituales se reportan métricas ordinales (MAE ordinal, accuracy ±1 estrato, kappa cuadrático).
# - Los datos tienen **coordenadas**: viviendas cercanas se parecen, y eso obliga a validar separando zonas.
#
# ¿Para qué serviría un modelo así? Para detectar viviendas cuyo estrato asignado no es coherente con sus
# características (posibles errores o desactualizaciones), para apoyar la revisión periódica de la estratificación y
# para estimar el estrato donde no está registrado. No reemplaza la estratificación oficial, que usa información que el
# catastro abierto no publica.
#
# ## 1.2 Cómo asigna el DANE el estrato
#
# La metodología vigente de estratificación urbana del DANE {cite}`dane2015conceptual,dane2015manual` trabaja, desde
# 2015, con la **información predial del catastro**. Tiene tres pasos:
#
# 1. **Zonas.** Se cruzan las zonas homogéneas físicas del catastro (topografía, vías, servicios, uso del suelo) con
#    las zonas homogéneas geoeconómicas (zonas de valor del suelo similar). Cada intersección es una *unidad espacial de
#    estratificación* (UEE).
# 2. **Estrato de la zona.** Cada UEE se califica de 0 a 100 con una medida de tendencia central del *puntaje de
#    calificación de las edificaciones* (estructura, acabados, baño y cocina). Los cortes entre estratos se calculan
#    para cada municipio de modo que la varianza dentro de los grupos sea mínima. Todas las viviendas de la UEE reciben
#    su estrato.
# 3. **Viviendas atípicas.** Solo cambia el estrato de una vivienda cuyo puntaje es atípico frente a las de su estrato:
#    fuera de $Q_1 - 1.5\,RIC$ o $Q_3 + 1.5\,RIC$ sube o baja un estrato; fuera de $3\,RIC$, dos.
#
# Lo que la metodología **no** usa: los avalúos catastrales (el DANE dice que trabaja «sobre las características físicas
# de las viviendas y su localización») ni la información del hogar, como el ingreso, la ocupación o la educación. Las
# zonas geoeconómicas sí reflejan el valor del suelo de la **zona**, no el de cada vivienda.

# %% tags=["hide-input"]
pd.DataFrame([
    ["Zonas homogéneas físicas y geoeconómicas", "Delimitar la zona (UEE)", "No; se aproximan con las coordenadas y la vecindad"],
    ["Puntaje de calificación de la edificación", "Calificar la zona y detectar atípicas", "No"],
    ["Área privada y cuartos (apartamentos)", "Calificar apartamentos", "Sí (área construida y habitaciones)"],
    ["Tipo de vivienda (casa o apartamento)", "Se califican por separado", "Sí"],
    ["Revisión de la alcaldía y el comité", "Ajustes manuales del resultado", "No observable"],
    ["Avalúo catastral", "No se usa (Ley 14 de 1983)", "Tampoco se usa"],
], columns=["Insumo del DANE", "Para qué", "¿Está en los datos del proyecto?"])

# %% [markdown]
# **Qué implica.** El estrato es, ante todo, una propiedad de la **zona**, y la vivienda solo se aparta de su zona
# cuando es claramente distinta. Por eso la pregunta del proyecto se precisa así: **¿cuánto del estrato se puede
# recuperar con la parte pública del catastro?** Esto anticipa tres resultados del informe: la ubicación será la
# información dominante, el estrato casi no variará dentro de un edificio, y los errores serán sobre todo de un
# estrato, el mismo salto que la metodología permite a las viviendas atípicas.
#
# ## 1.3 Revisión bibliográfica
#
# **Método.** Juan Camilo Oñoro realizó una revisión sistemática en **Scopus** (exportación del 5 de octubre de 2026)
# con dos ecuaciones de búsqueda:
#
# | Búsqueda | Pregunta | Para qué |
# |---|---|---|
# | 1. Aplicación | ¿Quién ha clasificado el nivel socioeconómico con aprendizaje supervisado a partir de datos catastrales o geoespaciales? | Saber si el problema ya está resuelto |
# | 2. Métodos | ¿Qué modelos de clasificación supervisada mejorados (optimización de hiperparámetros, selección de variables, *ensembles*, efectos espaciales) reducen el error frente a los clasificadores convencionales? | Encontrar candidatos para superar al modelo base |
#
# Se excluyeron el *deep learning*, las redes neuronales y el aprendizaje no supervisado, por el tamaño efectivo de la
# muestra (unas 2 300 observaciones independientes, capítulo 2) y por interpretabilidad.
#
# **Hallazgo de la búsqueda 1: no hay un antecedente directo.** No se encontró ningún estudio que clasifique el estrato
# o nivel socioeconómico de una **vivienda** usando como predictores sus características catastrales y su ubicación.
# Los 9 trabajos más cercanos (Tabla II de la revisión) predicen otra cosa:

# %% tags=["hide-input"]
pd.DataFrame([
    ["Precio o valorización de la vivienda", "5 de 9", "XGBoost, Random Forest, SVR, árbol, Gradient Boosting, lineal",
     "Sí trabajan vivienda por vivienda y con atributos físicos parecidos, pero el objetivo es un precio continuo "
     "(regresión, R² ≈ 0.88–0.89), y el estrato no puede basarse en el avalúo"],
    ["Pobreza multidimensional por zona (México)", "1 de 9", "XGBoost con datos censales",
     "Mide condiciones del hogar agregadas en celdas de 469 m: se parece más al SISBÉN (que clasifica hogares) que al "
     "estrato (que clasifica inmuebles)"],
    ["Nivel socioeconómico como predictor (salud, PISA, valorización)", "3 de 9", "Logística; RF, XGBoost, LASSO, stacking",
     "El nivel socioeconómico es una variable de entrada, no lo que se predice"],
    ["Elección de dónde vivir (1998)", "1 de 9", "Análisis factorial y regresión logística",
     "Datos agregados por sector censal; es el antecedente más cercano en método, pero no predice el estrato"],
], columns=["Qué predicen", "Estudios", "Modelos", "En qué se parece y en qué no"])

# %% [markdown]
# Referencias de la tabla: precio y valorización {cite}`rojas2026,ren2024,jin2024`; pobreza {cite}`salas2026`; elección
# residencial {cite}`sermons1998`. Uno de los estudios de valorización también usa el nivel socioeconómico como predictor.
#
# **Por qué no hay estudios comparables.** Las diferencias con este proyecto son tres:
#
# 1. **Objetivo.** El estrato solo existe en Colombia (sección 1.1); la literatura internacional predice el precio de la
#    vivienda, la pobreza o un índice socioeconómico del hogar.
# 2. **Unidad.** Los estudios socioeconómicos trabajan con zonas (celdas, sectores censales), donde los promedios
#    facilitan la predicción; los que trabajan con viviendas individuales predicen su precio.
# 3. **Tarea y métrica.** Casi todos son regresiones evaluadas con R²; aquí es una clasificación en 6 clases
#    ordenadas, evaluada con F1 macro.
#
# **Confirmación en Web of Science.** Una búsqueda complementaria en Web of Science Core Collection (19 de octubre de
# 2026), hecha directamente desde *Advanced Search*, con la consulta `TS=(("socioeconomic status" OR "socioeconomic
# stratum" OR "socioeconomic stratification") AND ("cadastral data" OR "housing characteristics"))`, no devolvió ningún
# documento. El dashboard muestra la captura (pestaña 1).
#
# **Alcance de este hallazgo.** «No se encontró» se refiere a esas dos bases (Scopus: dos ecuaciones, 5 de octubre de
# 2026, sin *deep learning*; Web of Science: la consulta anterior). Ambas indexan poca literatura colombiana en
# español: una versión final de la revisión debería sumar SciELO, Redalyc y repositorios de tesis colombianas, y el
# libro *Los límites de la estratificación* (Sepúlveda, Gallego y López), que propone usar información catastral como
# alternativa al estrato.
#
# **Hallazgo de la búsqueda 2: modelos mejorados que no se han aplicado a este problema.** En el orden de prioridad de
# la revisión:

# %% tags=["hide-input"]
pd.DataFrame([
    ["1", "**XGBoost geográfico (G-XGBoost)**", "Un XGBoost global más XGBoost locales con pesos espaciales; mejora R² y MAE ≈ 17 % frente a XGBoost, GWR y RF geográfico (regresión)", "El estrato cambia de forma local por barrios", "grekousis2025"],
    ["2", "GWO-XGBoost / TPE-SVM", "Optimización metaheurística o bayesiana de hiperparámetros", "Espacio de hiperparámetros amplio", "ojekemi2026, rizkallah2025"],
    ["3", "Ensemble RF + SVM", "RF selecciona variables y SVM clasifica", "Variables catastrales correlacionadas", "zhai2026"],
    ["4", "Bagged TAO Trees", "Árboles oblicuos optimizados; superan a RF, AdaBoost y GB", "Datos tabulares heterogéneos", "carreira2020"],
    ["5", "Rotation Forest", "Rotación PCA antes de cada árbol", "Colinealidad entre variables de tamaño", "sartono2018"],
], columns=["Prioridad", "Modelo", "Idea y evidencia", "Por qué encaja con este problema", "Referencia"])

# %% [markdown]
# ## 1.4 La brecha, el modelo propuesto y la hipótesis
#
# **La brecha tiene dos partes** (la «brecha doble» de la revisión):
#
# - **De aplicación.** Ningún trabajo clasifica el estrato de cada vivienda con sus características catastrales y su
#   ubicación, como problema de 6 clases.
# - **De método.** Los nueve modelos de la Tabla III reducen el error frente a los clasificadores convencionales en
#   otros problemas, pero nunca se aplicaron a este. El mejor respaldado para problemas espaciales, el **XGBoost
#   geográfico** de Grekousis {cite}`grekousis2025`, solo existe para **regresión**: no encontramos una versión para
#   clasificación ni una aplicación al estrato.
#
# **El matiz que precisa la brecha.** El DANE sí calcula el estrato con el catastro (sección 1.2), pero con insumos que
# el **catastro abierto no publica**: el puntaje de calificación de las edificaciones y las zonas homogéneas. El vacío no
# es que nadie use el catastro, sino que nadie ha medido **cuánto del estrato se recupera con la parte pública del
# catastro y aprendizaje supervisado**.
#
# **Consecuencia.** Como no hay cifras publicadas con las que compararse, el desempeño se juzga contra un **modelo base
# propio** (capítulo 3), evaluado con el mismo protocolo espacial que los modelos propuestos (capítulo 5).
#
# **El modelo propuesto.** Una **adaptación propia del XGBoost geográfico a clasificación multiclase** (capítulo 4):
# un XGBoost global más modelos locales alrededor de puntos ancla, con un kernel espacial adaptativo, que mezcla las
# probabilidades del global y de los locales. Encaja con el problema porque el estrato es una propiedad de la zona
# (sección 1.2) y un modelo con efectos locales puede aprender relaciones que cambian de un barrio a otro.
#
# ```{admonition} Objetivo e hipótesis del entregable
# :class: important
# **Objetivo.** Usar **modelos que no se habían aplicado a esta aplicación** (los candidatos de la revisión),
# encabezados por el XGBoost geográfico adaptado a clasificación, y comprobar si alguno es **mejor que el modelo
# base**, una regresión logística multinomial con las variables físicas y de ubicación de la vivienda (capítulo 3).
#
# **Hipótesis.** El XGBoost geográfico, el modelo principal, supera al modelo base en F1 macro con el mismo protocolo
# de evaluación (capítulo 5). Los demás candidatos originales de la revisión (Rotation Forest, SVM con TPE y, en la
# versión 3, RF + SVM y stacking) se evalúan con el mismo protocolo, y XGBoost sirve de control del modelo principal
# (capítulos 6, 8 y 9).
#
# **Criterio, fijado antes de ver resultados.** Un modelo reemplaza al base solo si su F1 macro en validación cruzada
# lo supera en más de un error estándar de la diferencia pareada por pliegue, y el test (que se mira una sola vez)
# debe confirmarlo.
# ```
#
# ## 1.5 Cómo se responde en este libro
#
# | Parte | Capítulos | Qué hace |
# |---|---|---|
# | I. Problema y datos | 1 (este) y 2 | La pregunta, la revisión y los datos |
# | II. Método | 3, 4 y 5 | El modelo base, el modelo propuesto y el protocolo con que se comparan |
# | III. Resultados | 6, 7 y 8 | Ronda 1 (variables del modelo base), ronda 2 (información del entorno) y la respuesta: ¿supera el modelo nuevo al base? |
# | Cierre | 9 y apéndices | Conclusiones; historia de las versiones, el dashboard y las referencias |
#
# **Respuesta corta (capítulo 8).** **Sí**, con matices: con las mismas variables del modelo base, el XGBoost
# geográfico lo supera en validación cruzada pero no se confirma en el test; cuando además recibe información del
# entorno de la vivienda (la **vecindad física**), lo supera en validación y en test en las zonas nuevas, y por mucho en
# las zonas con datos cerca.
