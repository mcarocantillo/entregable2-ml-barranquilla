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
# # 3. Modelos base
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# Este capítulo corresponde a la **pestaña 3 del dashboard**. El enunciado pide un modelo base lineal (regresión
# logística o SVR lineal); como el estrato es una clase, se usa **regresión logística**, comparada contra líneas base
# triviales.

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
print(f"Entrenamiento tras el buffer: {R['n_train']:,} viviendas (de {R['n_train_antes_buffer']:,}) | "
      f"Test: {R['n_test']:,} viviendas en {R['bloques_test']} bloques de 2 km".replace(",", " "))

# %% [markdown]
# ## 3.1 Los modelos
#
# **Modelo base: regresión logística multinomial.** Estima, para cada estrato $k$, una ecuación lineal
# $z_k = \beta_{0k} + \boldsymbol\beta_k^\top \mathbf x$ y convierte los seis puntajes en probabilidades con la función
# softmax, $P(y = k \mid \mathbf x) = e^{z_k} / \sum_j e^{z_j}$. Se predice el estrato con mayor probabilidad. Se usa
# penalización L2: el hiperparámetro $C$ es el inverso de la fuerza de regularización (un $C$ pequeño encoge más los
# coeficientes).
#
# **Pipeline.** Todo el preprocesamiento va dentro de un `Pipeline` de scikit-learn y se ajusta solo con los datos de
# entrenamiento de cada fold:
#
# | Paso | Variables | Hallazgo del EDA que lo motiva |
# |---|---|---|
# | Winsorización en p0.1 y p99.9 | numéricas | colas largas; extremos reales que no se eliminan |
# | Imputación por mediana + indicador de faltante | numéricas | < 0.2 % de faltantes, mecanismo MAR |
# | log(1 + x) | variables de tamaño, pisos y antigüedad | asimetría > 1 |
# | Estandarización | numéricas | escalas distintas; la penalización L2 lo requiere |
# | Categoría "faltante" + one-hot (categorías < 1 % agrupadas) | régimen, uso, tipo de vivienda | categorías raras |
#
# **Dos conjuntos de variables.** **A**: solo variables físicas y categóricas de la vivienda. **B**: A más la ubicación
# (x, y en km respecto al centro y distancia al centro). Comparar A y B mide cuánto aporta la ubicación. No se usa
# ningún rezago espacial del estrato (el estrato de los vecinos), porque sería fuga: en una zona nueva no se conoce.
#
# **Líneas base.** Tres `DummyClassifier` (clase mayoritaria, estratificada y uniforme) y la **moda por zona**: la
# clase más frecuente de la celda de 2 km en entrenamiento. Es la versión para clasificación de la "media por zona" que
# sugiere el enunciado para datos espaciales, y es exigente porque explota la segregación. El tamaño de zona (2 km) se
# eligió por validación cruzada entre varios candidatos.
#
# **Selección de hiperparámetros.** `GridSearchCV` con $C \in \{0.01, 0.1, 1, 10\}$ y pesos de clase
# $\in \{\text{ninguno}, \text{balanceados}\}$, optimizando el F1 macro con los mismos folds espaciales con buffer. Se
# aplicó la **regla de una desviación estándar**: entre las combinaciones cuyo F1 medio está a menos de un error
# estándar de la mejor, se elige la más regularizada {cite}`hastie2009`. Motivo: en la CV espacial las diferencias entre
# combinaciones (milésimas) son menores que la variabilidad entre folds.

# %% tags=["hide-input"]
hp = pd.DataFrame(R["mejores_hiperparametros"]).T.rename(columns={"clf__C": "C", "clf__class_weight": "Pesos de clase"})
hp.index = ["Logística A (físicas)", "Logística B (físicas + ubicación)"]
hp

# %% [markdown]
# ## 3.2 Comparación con las líneas base

# %% tags=["hide-input"]
F.fig_comparacion("f1_macro")

# %% tags=["hide-input"]
F.tabla_metricas()

# %% [markdown]
# **Lectura de la figura y la tabla.** Las barras con textura son la media de la validación cruzada espacial (con su
# desviación entre folds) y las de color son el resultado en test. La logística **B** tiene el mayor F1 macro tanto en
# CV (0.299) como en test (0.450), y por eso es el **modelo principal**: la elección se hizo con la CV, nunca con test.
#
# - Frente a las **Dummy**, B gana en todas las métricas y por mucho. La Dummy mayoritaria tiene accuracy 0.29 y F1
#   macro 0.07: acierta solo la clase que predice siempre.
# - Frente a la **moda por zona**, B es mejor en todas las métricas de test, en especial en las ordinales: MAE 0.46
#   frente a 0.94 y kappa cuadrático 0.81 frente a 0.43. La moda por zona acierta bien la clase en zonas homogéneas,
#   pero cuando se equivoca lo hace por varios estratos, y en zonas sin entrenamiento cercano recurre a la distribución
#   global.
# - Las métricas **ordinales** muestran que los errores de B son casi siempre de un estrato: el 94 % de las predicciones
#   cae en el estrato real o en uno vecino.
# - El **log loss** de las Dummy mayoritaria y estratificada es enorme porque asignan probabilidad 0 a clases que sí
#   aparecen; la Dummy uniforme (1/6 a todo) tiene el log loss de referencia, 1.79. B baja a 1.03.

# %% [markdown]
# ### Incertidumbre: intervalos por bloques
#
# Las viviendas del test no son independientes (sección 2.5), así que los intervalos de confianza se obtienen con
# **bootstrap por bloques**: se remuestrean con reemplazo los 7 bloques de test (500 réplicas) y se recalcula la métrica.

# %% tags=["hide-input"]
ic = pd.DataFrame(R["ic_modelo_principal"])
ic.columns = ["Valor en test", "IC 95 % inferior", "IC 95 % superior", "Desv. bootstrap"]
ic.index = ["Accuracy", "F1 macro", "MAE ordinal", "Kappa cuadrático"]
ic

# %% [markdown]
# Los intervalos son **muy anchos** (F1 macro entre 0.16 y 0.46) porque hay solo 7 bloques de test, y el valor puntual
# está cerca del extremo superior. La asimetría tiene una causa concreta: cuando una réplica no incluye los bloques
# que concentran los estratos 5 o 6, el F1 de esa clase cae a 0 y arrastra el promedio macro. Además, el F1 de test
# (0.45) supera al de la CV (0.30) por dos razones probables: los 7 bloques de test son favorables (solo 1.3 % de
# estrato 6, frente a 5.7 % en entrenamiento) y el modelo final se entrena con 138 239 viviendas, entre 1.9 y 2.6
# veces las de cada fold. Por eso la estimación más representativa
# del desempeño en zonas nuevas es la de la **CV espacial (F1 0.30 ± 0.09)**, que promedia cinco particiones.

# %% tags=["hide-input"]
comp = pd.DataFrame(R["comparaciones_f1"])
comp.columns = ["Diferencia de F1", "IC 95 % inf.", "IC 95 % sup.", "p (bootstrap)", "¿El IC excluye 0?"]
comp.index = [i.replace("Logística B_fisicas+ubicacion", "B").replace("Logística A_fisicas", "A") for i in comp.index]
comp

# %% [markdown]
# **Comparaciones pareadas.** Para comparar dos modelos se remuestrean los mismos bloques para ambos y se calcula la
# diferencia de F1 en cada réplica. B supera a las Dummy mayoritaria y estratificada con intervalos que excluyen el cero.
# Frente a la moda por zona (+0.12, IC [−0.01, 0.19]) y frente a A (+0.10, IC [−0.05, 0.14]) la diferencia es positiva
# pero **no significativa** al 95 %. Con tan pocos bloques de test no se puede afirmar con seguridad que B sea mejor que
# una regla geográfica gruesa en F1. En las métricas ordinales la ventaja es grande (MAE 0.46 frente a 0.94), aunque
# esa diferencia no se evaluó con un intervalo pareado.

# %% [markdown]
# ## 3.3 Por qué la validación tiene que ser espacial

# %% tags=["hide-input"]
F.fig_optimismo()

# %% tags=["hide-input"]
sens = pd.DataFrame(D["sensibilidad_buffer"]).rename(columns={"buffer_km": "Buffer (km)", "f1": "F1 macro CV",
                                                              "desv": "Desv. entre folds"})
sens

# %% [markdown]
# Con el mismo modelo B, una validación cruzada **aleatoria** (filas mezcladas, `StratifiedKFold`) da un F1 macro de
# **0.60**, y la validación **espacial** con bloques y buffer, **0.30**: el doble. La diferencia se debe a que, al
# mezclar filas, apartamentos de la misma torre y casas de la misma cuadra quedan a la vez en entrenamiento y
# validación, y el modelo "reconoce" la zona. Ese 0.60 describiría qué tan bien el modelo interpola dentro de zonas que
# ya vio, no qué tan bien generaliza a zonas nuevas.
#
# El **buffer** también cambia la cifra: sin buffer, el F1 de CV es 0.40; con 0.5 km, 0.35; con 1 km, 0.30; con el
# alcance del correlograma (2.9 km), 0.18. A partir de cierto punto la caída mezcla dos efectos: menos fuga por cercanía
# y menos datos de entrenamiento, porque el buffer elimina viviendas. El buffer de 1 km es un compromiso entre los dos.

# %% [markdown]
# ## 3.4 Diagnóstico del modelo principal

# %% tags=["hide-input"]
F.fig_confusion("Logística B_fisicas+ubicacion", normalizar=True)

# %% tags=["hide-input"]
F.tabla_por_clase("Logística B_fisicas+ubicacion")

# %% [markdown]
# **Matriz de confusión** (cada fila suma 100 %). Los aciertos se concentran en los estratos 1 (87 %), 3 (88 %) y 4
# (60 %). Hay dos desplazamientos sistemáticos **hacia el centro de la escala**:
#
# - El **estrato 2 casi nunca se predice**: solo el 6 % se clasifica bien y el 83 % se clasifica como 3. Su F1 es 0.11.
# - Los estratos **5 y 6** se predicen como 4 en el 56 % de los casos. El estrato 6 tiene precisión 0.38 y recall 0.21.
#
# Casi no hay errores de más de un estrato (por ejemplo, prácticamente ningún estrato 1 se predice como 5 o 6: solo 2 viviendas), lo que explica el
# accuracy ±1 de 0.94.

# %% tags=["hide-input"]
F.fig_roc("Logística B_fisicas+ubicacion")

# %% [markdown]
# **Curvas ROC** (uno contra el resto). Todas las clases tienen AUC entre 0.81 y 0.96, incluido el estrato 2 (0.86). Es
# decir, el modelo **ordena bien** las viviendas de estrato 2 (les asigna más probabilidad de ser 2 que a las demás),
# pero esa probabilidad casi nunca es la más alta: pierde el argmax frente al 3. El problema del estrato 2 no es falta
# de señal, sino de calibración y de umbral de decisión.

# %% tags=["hide-input"]
F.fig_calibracion("Logística B_fisicas+ubicacion")

# %% tags=["hide-input"]
y, pred, proba = F._pred("Logística B_fisicas+ubicacion")
conf = proba.max(1)
acierto = (pred == y).astype(float)
bins = np.quantile(conf, np.linspace(0, 1, 11))
idx = np.clip(np.digitize(conf, bins[1:-1]), 0, 9)
ece = sum(abs(acierto[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in range(10))
cal = pd.DataFrame({"Estrato": F.CLASES, "Probabilidad media predicha": proba.mean(0),
                    "Frecuencia observada en test": [(y == k).mean() for k in F.CLASES]})
print(f"Error de calibración esperado (ECE, 10 bins por cuantiles de la confianza): {ece:.3f}")
cal

# %% [markdown]
# **Calibración.** En promedio, el modelo asigna al estrato 3 una probabilidad de 0.41, cuando en test solo el 22 % de
# las viviendas son de estrato 3: lo **sobreestima**. Al estrato 2 le asigna 0.09 frente a un 21 % real: lo
# **subestima**. En la figura, la curva del 3 queda por debajo de la diagonal y las del 1 y el 2 por encima. El error
# de calibración esperado (ECE) es 0.14, el mismo valor del Entregable 1: las probabilidades son orientativas, no
# exactas. Parte de este
# sesgo viene del cambio de zona: el test tiene una composición de estratos distinta a la del entrenamiento que queda
# tras el buffer.

# %% [markdown]
# ### Comparación con las líneas base

# %% tags=["hide-input"]
F.fig_confusion("Moda por zona", normalizar=True)

# %% [markdown]
# La **moda por zona** casi nunca predice el estrato 2 (103 viviendas) y nunca el 5. Acierta casi todo el
# estrato 1 (99 %), pero asigna estrato 1 al 63 % del estrato 2 y al 65 % del estrato 5. La razón es que muchas celdas
# de test no tienen entrenamiento cercano (por el buffer) y la regla recurre a la clase global, el estrato 1; y en
# celdas de 2 km conviven estratos distintos, todos con la misma predicción. La logística, al combinar la ubicación
# continua con las características de la vivienda, evita la mayoría de esos saltos: por eso su MAE ordinal es la mitad.

# %% [markdown]
# ## 3.5 Interpretación de los coeficientes

# %% tags=["hide-input"]
F.fig_coeficientes("B_fisicas+ubicacion")

# %% [markdown]
# Cada celda es el coeficiente $\beta$ de una variable en la ecuación de un estrato. Las numéricas están estandarizadas
# (después del logaritmo, si aplica), así que $\beta$ es el cambio en el puntaje de ese estrato por cada desviación
# estándar de la variable. En la multinomial lo que importa son las **diferencias** entre estratos: $e^{\beta_6 - \beta_1}$
# es cuánto cambia la razón de probabilidades entre el estrato 6 y el 1 por una desviación estándar.
#
# - **Posición norte-sur (y)** es la variable dominante: $\beta$ va de −3.17 en el estrato 1 a +1.94 en el 6 (el máximo, +2.13, está en el 4,
#   coherente con que la franja más al norte no sea la de estrato más alto). Moverse
#   una desviación estándar hacia el norte multiplica la razón de probabilidades "estrato 6 frente a 1" por
#   $e^{1.94 + 3.17} \approx 166$.
# - Entre las físicas, el **área construida** y los **baños** empujan hacia estratos altos ($\beta_6 - \beta_1$ = 3.4 y
#   1.4), y el régimen **informal** hacia el estrato 1.
# - La **distancia al centro** y la **posición este-oeste** tienen coeficientes no monótonos (por ejemplo, la distancia
#   favorece los estratos 1 y 2 y penaliza el 4): capturan, de forma gruesa, que los estratos medios se concentran en
#   ciertos sectores y no en una línea recta.
#
# **Cautelas.** Las variables están correlacionadas, así que un coeficiente aislado no es un efecto causal ni puede
# leerse "con todo lo demás constante" en sentido estricto. Además, la multinomial no usa el orden de los estratos: una
# logística ordinal sería más parsimoniosa.

# %% tags=["hide-input"]
F.fig_coeficientes("A_fisicas")

# %% [markdown]
# Sin ubicación (modelo **A**), el peso pasa a las variables físicas y al régimen de propiedad: los predios informales y
# el área construida separan los extremos de la escala. A llega a un F1 de 0.28 en CV (frente a 0.30 de B) y de 0.35 en
# test (frente a 0.45). Que la diferencia en CV sea tan pequeña indica que buena parte de la información de la ubicación
# ya está "contenida" en las características físicas (las casas grandes con varios baños están en el norte).

# %% [markdown]
# ## 3.6 Curva de aprendizaje

# %% tags=["hide-input"]
F.fig_curva_aprendizaje()

# %% tags=["hide-input"]
ca = D["curva_aprendizaje"]
pd.DataFrame({"Viviendas de entrenamiento": ca["n"], "F1 entrenamiento": ca["train_media"],
              "F1 validación espacial": ca["val_media"],
              "Brecha": np.array(ca["train_media"]) - np.array(ca["val_media"])})

# %% [markdown]
# Al multiplicar por 20 el tamaño del entrenamiento (de 2 700 a 54 000 viviendas, el tamaño del fold más pequeño), el F1
# de validación espacial solo pasa de 0.27 a 0.30. **Es poco probable que más filas cambien mucho este modelo**, aunque
# la curva no llega a las 138 239 viviendas del modelo final. La brecha con el F1 de entrenamiento (0.48 a 0.59) no es el sobreajuste clásico de un
# modelo que memoriza filas: es la diferencia entre predecir en zonas conocidas y en zonas nuevas. Y el F1 de
# entrenamiento *sube* con n, porque con más zonas el modelo aprende mejor el gradiente espacial. El modelo está
# limitado por su forma (lineal, sin interacciones) y por la información disponible, no por la cantidad de filas.

# %% [markdown]
# ## 3.7 Residuos espaciales

# %% tags=["hide-input"]
mr = pd.DataFrame(D["moran_residuos"]).T.loc[["A_fisicas", "B_fisicas+ubicacion"]]
mr.columns = ["I de Moran de los residuos", "p (permutación)", "Residuo medio"]
mr.index = ["Logística A (físicas)", "Logística B (físicas + ubicación)"]
mr

# %% tags=["hide-input"]
F.fig_mapa_residuos()

# %% [markdown]
# El residuo ordinal es $y - \mathrm{E}[y]$, con $\mathrm{E}[y] = \sum_k k\,P(y = k)$: usa toda la distribución
# predicha y respeta el orden. Se calcula por edificio de test y se mide su autocorrelación con el I de Moran (8 vecinos).
#
# Los residuos de B tienen un **Moran de 0.70**: siguen muy autocorrelacionados. El modelo capta el gradiente de gran
# escala (norte-sur), pero no los barrios: en el mapa hay zonas enteras en azul (estrato real mayor que el esperado) y
# zonas enteras en rojo (menor). El residuo medio es negativo (−0.14): en conjunto el modelo sobreestima ligeramente el
# estrato del test. Paradójicamente, A tiene residuos menos autocorrelacionados (0.34) pero mucho mayores en magnitud.
# Al no ver la ubicación, A se equivoca en todas partes y de forma más dispersa; B corrige la tendencia general y deja
# solo el componente local.
#
# **Implicación para el siguiente entregable.** Los residuos con estructura espacial indican información que el modelo
# no está usando. Es justo el caso que motiva los modelos con **efectos espaciales locales** de la revisión
# bibliográfica, como el XGBoost geográfico {cite}`grekousis2025`, y la construcción de variables de vecindad física
# (por ejemplo, el área o los baños promedio de los vecinos, que no usan el estrato y no son fuga).

# %% [markdown]
# ## 3.8 Simulador
#
# La última sección del dashboard permite cambiar las características de una vivienda y su ubicación y ver las
# probabilidades que asigna el modelo B. Usa el mismo modelo ajustado (guardado con `joblib`) y proyecta las coordenadas
# con la misma función del Entregable 1. Dos ejemplos con una vivienda típica (las medianas de entrenamiento) en dos
# ubicaciones:

# %% tags=["hide-input"]
rg = F.rangos_simulador()
base = {c: rg[c][1] for c in R["variables_modelo"]["numericas"]}
base.update(rg["moda_categorias"])
# Tres celdas reales del mapa (con al menos 300 viviendas) sobre la misma longitud: la de estrato medio más bajo, una
# intermedia y la de estrato medio más alto.
celdas = F._grilla(F.viviendas(), "estrato_num", min_n=300).sort_values("valor")
puntos = celdas.iloc[[0, len(celdas) // 2, -1]]
filas = []
for nombre, (_, c) in zip(["Celda de estrato medio más bajo", "Celda intermedia", "Celda de estrato medio más alto"],
                          puntos.iterrows()):
    p = F.predecir(dict(base, lat=float(c.glat), lon=float(c.glon)))
    filas.append({"Ubicación": nombre, "lat": round(c.glat, 3), "lon": round(c.glon, 3),
                  "Estrato medio real de la celda": round(c.valor, 2),
                  **{f"P(E{k})": round(p[k], 3) for k in F.CLASES}, "Estrato predicho": max(p, key=p.get)})
pd.DataFrame(filas)

# %% [markdown]
# Se usa la misma vivienda típica (las medianas de entrenamiento: unos 69 m², 3 habitaciones y 1 baño, es decir, una
# vivienda modesta) en tres celdas reales. En la celda de estrato 1, el modelo la predice como estrato 1 con 93 % de
# probabilidad. En la celda de estrato 6, **no** la predice como 6, sino como 3: la ubicación sube la probabilidad de
# los estratos medios y altos, pero las características de una vivienda pequeña con un baño la frenan. El modelo
# combina las dos fuentes de información, y lo hace de forma aditiva y lineal: no puede representar que "en este
# barrio todas las viviendas son estrato 6", que es justo lo que un modelo con efectos locales debería capturar.
# En la celda intermedia (estrato medio real 2.6) duda entre 1 (43 %) y 3 (32 %), y predice 1.
#
# El simulador es una herramienta para **entender** el modelo, no para asignar estratos reales.

# %% [markdown]
# ## 3.9 Nota crítica del enunciado
#
# El enunciado pide revisar fuga de datos si la accuracy supera 0.80–0.90. La accuracy del modelo principal en test es
# **0.60** y en la CV espacial 0.46, lejos de ese umbral. Aun así, la auditoría de fuga del Entregable 1 descartó el
# número predial y cualquier rezago espacial del estrato, la partición es por bloques con buffer, y el preprocesamiento
# se aprende dentro del Pipeline. El problema no es trivial: una regla geográfica gruesa ya alcanza buena parte del
# desempeño, y lo que queda por ganar está en los barrios y en los estratos intermedios.
