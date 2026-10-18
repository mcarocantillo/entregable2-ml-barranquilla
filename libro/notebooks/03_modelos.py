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
# Este capítulo corresponde a la pestaña **4. Modelo base** del dashboard (protocolo, hiperparámetros, comparación con las líneas base, diagnóstico, coeficientes, curva de aprendizaje, residuos y simulador). El enunciado pide un modelo base lineal (regresión
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
# clase más frecuente de la celda de 4 km en entrenamiento. Es la versión para clasificación de la "media por zona" que
# sugiere el enunciado para datos espaciales, y es exigente porque explota la segregación. El tamaño de zona (4 km) se
# eligió por validación cruzada entre varios candidatos (2, 4, 6 y 8 km).
#
# **Selección de hiperparámetros.** `GridSearchCV` con $C \in \{0.01, 0.1, 1, 10\}$ y pesos de clase
# $\in \{\text{ninguno}, \text{balanceados}\}$, optimizando el F1 macro con los mismos folds espaciales con buffer. Se
# aplicó la **regla de una desviación estándar**: entre las combinaciones cuyo F1 medio está a menos de un error
# estándar de la mejor, se elige la más regularizada {cite}`hastie2009`. Motivo: en la CV espacial las diferencias entre
# combinaciones (centésimas) son menores que la variabilidad entre folds. Así quedaron A con C = 0.01 sin pesos y B con
# C = 0.01 y pesos balanceados.

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
# CV (0.322) como en test (0.492), y por eso es el **modelo principal**: la elección se hizo con la CV, nunca con test.
#
# - Frente a las **Dummy**, B gana en todas las métricas y por mucho. La Dummy mayoritaria tiene accuracy 0.35 y F1
#   macro 0.09: acierta solo la clase que predice siempre.
# - Frente a la **moda por zona**, B es mejor en todas las métricas de test, en especial en las ordinales: MAE 0.44
#   frente a 0.52 y kappa cuadrático 0.84 frente a 0.77. La moda por zona acierta bien la clase en zonas homogéneas,
#   pero nunca predice los estratos 5 y 6, y en zonas sin entrenamiento cercano recurre a la distribución global.
# - Las métricas **ordinales** muestran que los errores de B son casi siempre de un estrato: el 96 % de las predicciones
#   cae en el estrato real o en uno vecino.
# - El **log loss** de las Dummy mayoritaria y estratificada es enorme porque asignan probabilidad 0 a clases que sí
#   aparecen; la Dummy uniforme (1/6 a todo) tiene el log loss de referencia, 1.79. B baja a 0.91.

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
# Los intervalos son **muy anchos** (F1 macro entre 0.28 y 0.51) porque hay solo 7 bloques de test, y el valor puntual
# está cerca del extremo superior. La asimetría tiene una causa concreta: cuando una réplica no incluye los bloques
# que concentran los estratos 5 o 6, el F1 de esa clase cae a 0 y arrastra el promedio macro. Además, el F1 de test
# (0.49) supera al de la CV (0.32) por dos razones probables: los 7 bloques de test son favorables (solo 1.2 % de
# estrato 6, frente a 4.8 % en entrenamiento) y el modelo final se entrena con 144 586 viviendas, entre 1.8 y 3.4
# veces las de cada fold. Por eso la estimación más representativa
# del desempeño en zonas nuevas es la de la **CV espacial (F1 0.32 ± 0.08)**, que promedia cinco particiones.

# %% tags=["hide-input"]
comp = pd.DataFrame(R["comparaciones_f1"])
comp.columns = ["Diferencia de F1", "IC 95 % inf.", "IC 95 % sup.", "p (bootstrap)", "¿El IC excluye 0?"]
comp.index = [i.replace("Logística B_fisicas+ubicacion", "B").replace("Logística A_fisicas", "A") for i in comp.index]
comp

# %% [markdown]
# **Comparaciones pareadas.** Para comparar dos modelos se remuestrean los mismos bloques para ambos y se calcula la
# diferencia de F1 en cada réplica. B supera a las Dummy mayoritaria y estratificada, y también a A (+0.12, IC
# [0.03, 0.15]), con intervalos que excluyen el cero. Frente a la moda por zona (+0.14, IC [−0.12, 0.17]) la diferencia
# es positiva pero **no significativa** al 95 %. Con tan pocos bloques de test no se puede afirmar con seguridad que B
# sea mejor que una regla geográfica gruesa en F1. En las métricas ordinales la ventaja también es clara (MAE 0.44
# frente a 0.52), aunque esa diferencia no se evaluó con un intervalo pareado.

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
# **0.62**, y la validación **espacial** con bloques y buffer, **0.32**: casi el doble. La diferencia se debe a que, al
# mezclar filas, apartamentos de la misma torre y casas de la misma cuadra quedan a la vez en entrenamiento y
# validación, y el modelo "reconoce" la zona. Ese 0.62 describiría qué tan bien el modelo interpola dentro de zonas que
# ya vio, no qué tan bien generaliza a zonas nuevas.
#
# El **buffer** también cambia la cifra: sin buffer, el F1 de CV es 0.41; con 0.5 km, 0.37; con 1 km, 0.32; con el
# alcance del correlograma (2.6 km), 0.15. A partir de cierto punto la caída mezcla dos efectos: menos fuga por cercanía
# y menos datos de entrenamiento, porque el buffer elimina viviendas. El buffer de 1 km es un compromiso entre los dos.

# %% [markdown]
# ## 3.4 Diagnóstico del modelo principal

# %% tags=["hide-input"]
F.fig_confusion("Logística B_fisicas+ubicacion", normalizar=True)

# %% tags=["hide-input"]
F.tabla_por_clase("Logística B_fisicas+ubicacion")

# %% [markdown]
# **Matriz de confusión** (cada fila suma 100 %). Los aciertos se concentran en los estratos 1 (79 %), 3 (70 %) y 2
# (60 %). Los problemas están en la mitad alta de la escala:
#
# - El **estrato 4 casi nunca gana**: solo el 28 % se clasifica bien; el 48 % se clasifica como 3 y el 22 % como 5. Su
#   F1 es 0.40.
# - Los estratos **5 y 6** se confunden entre sí y con el 4: el 5 acierta el 42 % y el 6 solo el 21 % (el 36 % del 6
#   se predice como 5 y el 30 % como 4). El estrato 6 tiene precisión 0.38.
# - Entre los estratos bajos el error típico es entre vecinos: el 18 % del estrato 1 se predice como 2 y el 24 % del 2
#   como 1.
#
# Casi no hay errores de más de un estrato (por ejemplo, prácticamente ningún estrato 1 se predice como 5 o 6: solo 3
# viviendas), lo que explica el accuracy ±1 de 0.96.

# %% tags=["hide-input"]
F.fig_roc("Logística B_fisicas+ubicacion")

# %% [markdown]
# **Curvas ROC** (uno contra el resto). Todas las clases tienen AUC entre 0.83 y 0.95, incluido el estrato 4 (0.93). Es
# decir, el modelo **ordena bien** las viviendas de estrato 4 (les asigna más probabilidad de ser 4 que a las demás),
# pero esa probabilidad casi nunca es la más alta: pierde el argmax frente al 3 o al 5. El problema del estrato 4 no es
# falta de señal, sino de calibración y de umbral de decisión. (En el modelo anterior, sin pesos de clase, ese papel lo
# tenía el estrato 2; con pesos balanceados el 2 se recupera.)

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
# **Calibración.** En promedio, el modelo asigna al estrato 4 una probabilidad de 0.10, cuando en test el 19 % de las
# viviendas son de estrato 4: lo **subestima**, y lo mismo hace con el 1 (0.29 frente a 0.35). En cambio,
# **sobreestima** el 2 (0.28 frente a 0.20) y el 3 (0.25 frente a 0.19). Es la huella de los pesos balanceados, que
# suben la probabilidad de las clases menos frecuentes en entrenamiento y bajan la del estrato 1. El error de
# calibración esperado (ECE) de la clase predicha es 0.03, el mismo valor del Entregable 1: en conjunto, la
# probabilidad de la clase predicha corresponde a su tasa de acierto, pero las probabilidades por clase están
# desplazadas. Parte de este sesgo viene también del cambio de zona: el test tiene una composición de estratos distinta
# a la del entrenamiento que queda tras el buffer.

# %% [markdown]
# ### Comparación con las líneas base

# %% tags=["hide-input"]
F.fig_confusion("Moda por zona", normalizar=True)

# %% [markdown]
# La **moda por zona** nunca predice los estratos 5 y 6, y casi nunca el 2 (3 076 viviendas). Acierta casi todo el
# estrato 1 (93 %), pero asigna estrato 1 al 51 % del estrato 2, estrato 4 al 72 % del estrato 5 y estrato 3 al 54 %
# del 6. La razón es que muchas celdas de test no tienen entrenamiento cercano (por el buffer) y la regla recurre a la
# clase global, el estrato 1; y en celdas de 4 km conviven estratos distintos, todos con la misma predicción. La
# logística, al combinar la ubicación continua con las características de la vivienda, evita la mayoría de esos saltos:
# por eso su MAE ordinal es menor (0.44 frente a 0.52).

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
# - **Posición norte-sur (y)** es la variable dominante: $\beta$ sube de forma monótona de −2.47 en el estrato 1 a
#   +3.64 en el 6. Moverse una desviación estándar hacia el norte multiplica la razón de probabilidades "estrato 6
#   frente a 1" por $e^{3.64 + 2.47} \approx 450$.
# - Entre las físicas, el **área construida** y los **baños** empujan hacia estratos altos ($\beta_6 - \beta_1$ = 2.9 y
#   1.4), y el régimen **informal** hacia los estratos 1 y 2.
# - La **distancia al centro** y la **posición este-oeste** tienen coeficientes no monótonos (por ejemplo, la distancia
#   favorece los estratos 1 y 2 y penaliza el 3, el 4 y el 5): capturan, de forma gruesa, que los estratos medios se
#   concentran en ciertos sectores y no en una línea recta.
#
# **Cautelas.** Las variables están correlacionadas, así que un coeficiente aislado no es un efecto causal ni puede
# leerse "con todo lo demás constante" en sentido estricto. Además, la multinomial no usa el orden de los estratos: una
# logística ordinal sería más parsimoniosa.

# %% tags=["hide-input"]
F.fig_coeficientes("A_fisicas")

# %% [markdown]
# Sin ubicación (modelo **A**), el peso pasa a las variables físicas y al régimen de propiedad: los predios informales y
# el área construida separan los extremos de la escala. A llega a un F1 de 0.29 en CV (frente a 0.32 de B) y de 0.37 en
# test (frente a 0.49). Que la diferencia en CV sea pequeña indica que buena parte de la información de la ubicación
# ya está "contenida" en las características físicas (las casas grandes con varios baños están en el norte, y las
# mejoras informales sin terreno, en el sur).

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
# Al multiplicar por 20 el tamaño del entrenamiento (de 4 000 a 80 000 viviendas, el tamaño del primer fold), el F1
# de validación espacial no sube: pasa de 0.33 a 0.32. **Es poco probable que más filas cambien este modelo**, aunque
# la curva no llega a las 144 586 viviendas del modelo final. La brecha con el F1 de entrenamiento (0.53 a 0.58) no es el sobreajuste clásico de un
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
# Los residuos de B tienen un **Moran de 0.56** (las viviendas ubicadas por su manzana cuentan como un punto por
# manzana): siguen muy autocorrelacionados. El modelo capta el gradiente de gran escala (norte-sur), pero no los
# barrios: en el mapa hay zonas enteras en azul (estrato real mayor que el esperado) y zonas enteras en rojo (menor).
# El residuo medio es prácticamente 0 (+0.003): los errores se compensan entre zonas. A tiene residuos menos
# autocorrelacionados (0.47) pero mayores en magnitud (MAE 0.71 frente a 0.44). Al no ver la ubicación, A se equivoca
# en todas partes y de forma más dispersa; B corrige la tendencia general y deja solo el componente local.
#
# **Implicación para el siguiente entregable.** Los residuos con estructura espacial indican información que el modelo
# no está usando. Es justo el caso que motiva los modelos con **efectos espaciales locales** de la revisión
# bibliográfica, como el XGBoost geográfico {cite}`grekousis2025`, y la construcción de variables de vecindad física
# (por ejemplo, el área o los baños promedio de los vecinos, que no usan el estrato y no son fuga).

# %% [markdown]
# ## 3.8 Simulador
#
# La última diapositiva de la pestaña Modelo base (4.11) permite cambiar las características de una vivienda y su ubicación y ver las
# probabilidades que asigna el modelo B. Usa el mismo modelo ajustado (guardado con `joblib`) y proyecta las coordenadas
# con la misma función del Entregable 1. Tres ejemplos con una vivienda típica (las medianas de entrenamiento) en tres
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
# Se usa la misma vivienda típica (las medianas de entrenamiento: unos 68 m², 3 habitaciones y 1 baño, y la
# combinación más frecuente de las categóricas, una casa NPH de hasta 3 pisos) en tres celdas reales. En la celda de
# estrato 1, el modelo la predice como estrato 1 con 65 % de probabilidad. En la celda intermedia (estrato medio real
# 2.3) la predice como 2 con 68 %. En la celda de estrato 6, **no** la predice como 6, sino como 2 (41 %), seguido del
# 1 (27 %) y el 3 (21 %): la ubicación sube la probabilidad de los estratos medios y altos, pero las características
# de una casa pequeña con un baño la frenan. El modelo combina las dos fuentes de información, y lo hace de forma
# aditiva y lineal: no puede representar que "en este barrio todas las viviendas son estrato 6", que es justo lo que un
# modelo con efectos locales debería capturar.
#
# El simulador es una herramienta para **entender** el modelo, no para asignar estratos reales.

# %% [markdown]
# ## 3.9 Nota crítica del enunciado
#
# El enunciado pide revisar fuga de datos si la accuracy supera 0.80–0.90. La accuracy del modelo principal en test es
# **0.61** y en la CV espacial 0.43, lejos de ese umbral. Aun así, la auditoría de fuga del Entregable 1 descartó el
# número predial y cualquier rezago espacial del estrato, la partición es por bloques con buffer, y el preprocesamiento
# se aprende dentro del Pipeline. El problema no es trivial: una regla geográfica gruesa ya alcanza buena parte del
# desempeño, y lo que queda por ganar está en los barrios y en los estratos intermedios.
