# 5. Variables de vecindad física y los dos escenarios de predicción

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

Este capítulo corresponde a la **sub-pestaña «Vecindad y escenarios»** de la pestaña **ML models** del dashboard.

Los capítulos 3 y 4 dejaron dos pistas. Los residuos de la logística siguen agrupados en el espacio (Moran = 0.70),
así que al modelo le falta información de **barrio**. Y los modelos más flexibles no ayudan en **zonas nuevas**, lo
que sugiere que el límite está en la información disponible y no en el algoritmo. Este capítulo pone a prueba esas
dos ideas con dos experimentos:

1. **Variables de vecindad física** en el escenario de siempre (zonas nuevas): ¿mejora la logística si, además de la
   vivienda, ve cómo son las viviendas que la rodean?
2. **El escenario de zonas conocidas**, que el capítulo 4 dejó pendiente: predecir viviendas ocultas dentro de zonas
   con datos, donde el XGBoost geográfico debería rendir mejor.

```{admonition} Resumen del capítulo
:class: important
- **Zonas nuevas.** Las variables de vecindad mejoran el test (F1 de 0.450 a 0.590), pero la validación cruzada no lo
  confirma: con 5 folds la mejora es de +0.025 y con 15 folds de +0.006, ambas dentro del ruido. El efecto **depende de
  la zona**: ayuda donde el barrio es homogéneo y perjudica donde está muy mezclado. Es un **resultado mixto**, no una
  mejora demostrada, y la logística B sigue siendo la referencia.
- **Zonas conocidas.** El orden se invierte: los árboles superan a la logística por +0.20 de F1. El **XGBoost
  geográfico supera a su control en los 5 folds** (+0.010, IC [+0.007, +0.024]): en su escenario natural, la
  adaptación sí aporta, aunque poco. Pero no supera a la regla más simple, votar con el estrato de las 15 viviendas
  conocidas más cercanas (F1 0.788 frente a 0.784, un empate).
```

## 5.1 Cómo se hizo: una réplica verificada del protocolo

El código está en la carpeta `vecindad/` del repositorio (ver su `README.md`). En el equipo donde se corrió, una
política de control de aplicaciones de Windows bloquea las librerías compiladas de scikit-learn. Por eso el
preprocesamiento y la regresión logística se **reimplementaron con numpy y scipy**, con las mismas fórmulas del
Entregable 1. La limpieza, la partición y los folds se tomaron sin cambios del código del Entregable 1.

Antes de usar la réplica se comprobó que reproduce las cifras guardadas del modelo B:

| | Réplica | Entregable 1 |
|---|---|---|
| Viviendas de entrenamiento / tras el buffer / de test | 218 870 / 138 239 / 59 954 | 218 870 / 138 239 / 59 954 |
| F1 macro por fold (CV espacial) | 0.421, 0.198, 0.379, 0.241, 0.256 | 0.420, 0.197, 0.379, 0.241, 0.257 |
| F1 macro en CV (media) | 0.299 | 0.299 |
| Combinación elegida por la regla 1-SE (B / A) | C = 0.01 sin pesos / C = 0.01 balanceado | igual |
| F1 macro en test | 0.4500 | 0.4500 |
| Accuracy / MAE ordinal / kappa en test | 0.598 / 0.463 / 0.814 | 0.598 / 0.464 / 0.814 |
| Moran de los residuos | 0.700 | 0.70 |

Las diferencias son de una milésima como máximo, así que las comparaciones de este capítulo usan el mismo protocolo
que el modelo base.

## 5.2 Las variables de vecindad

Para cada vivienda se resume cómo son las viviendas que la rodean, a dos escalas, **300 m y 1 km**:

| Variable | Qué mide |
|---|---|
| `area_log` | Área construida media de las vecinas (en logaritmo) |
| `banios` | Número medio de baños |
| `antiguedad` | Antigüedad media |
| `apto` | Proporción de apartamentos en edificios de cuatro pisos o más |
| `informal` | Proporción de predios informales |
| `dens` | Densidad: logaritmo del número de viviendas vecinas |

Son 12 variables en total (6 por escala). Tres reglas evitan la fuga de datos:

- **Solo atributos físicos.** Nunca se usa el estrato de los vecinos: en una zona nueva no se conoce. Estas variables
  están disponibles en cualquier lugar del catastro.
- **Se excluye el propio edificio.** Sus otras unidades repetirían la información de la vivienda.
- **No se aprende nada de los datos.** Es una regla geométrica fija. La imputación y el escalado siguen dentro del
  `Pipeline` y se ajustan solo con el entrenamiento de cada fold.

El cálculo agrupa las viviendas en una rejilla de 50 m y suma cada atributo dentro de un disco con una convolución.
Se verificó contra un cálculo exacto por fuerza bruta (`prueba_variables_vecindad.py`), y coincide en el 100 % de
las viviendas probadas. La mediana de viviendas vecinas es de 1 130 a 300 m y de 10 839 a 1 km, y solo el 0.4 % de
las viviendas no tiene vecinas a 300 m.

## 5.3 Zonas nuevas: el criterio, fijado antes de ver los resultados

- **Modelo principal:** la logística B más las 12 variables de vecindad. Las demás variantes (una sola escala, o sin
  coordenadas) son **exploratorias**.
- **Mismo protocolo que B:** los mismos 5 folds espaciales con buffer de 1 km, la misma rejilla de C y pesos de clase,
  la misma regla de una desviación estándar y el mismo test.
- **Criterio de reemplazo (el del capítulo 4):** un modelo reemplaza a B solo si su F1 de CV la supera en más de un
  error estándar de la diferencia pareada por fold.

## 5.4 Zonas nuevas: resultados

| Modelo | F1 CV | Δ CV (error est.) | ¿Reemplaza a B? | F1 test | Δ F1 test (IC 95 % por bloques) |
|---|---|---|---|---|---|
| Logística B (referencia) | 0.299 | — | — | 0.450 | — |
| **B + vecindad (300 m y 1 km)** | 0.324 | +0.025 (0.041) | No | **0.590** | **+0.140 [+0.040, +0.244]** |
| B + vecindad, solo 300 m | 0.310 | +0.011 (0.032) | No | 0.570 | +0.120 [+0.007, +0.157] |
| B + vecindad, solo 1 km | 0.336 | +0.037 (0.040) | No | 0.583 | +0.134 [+0.032, +0.171] |
| A: solo variables físicas | 0.283 | −0.016 (0.065) | No | 0.349 | −0.101 [−0.140, +0.054] |
| A + vecindad (sin coordenadas) | 0.356 | +0.056 (0.028) | Sí | 0.546 | +0.096 [−0.021, +0.218] |

En **test**, el modelo principal mejora todas las métricas: accuracy de 0.598 a 0.738, MAE ordinal de 0.463 a 0.281
(IC de la diferencia [−0.446, −0.026]) y kappa cuadrático de 0.814 a 0.906. El cambio más visible está en el estrato
2, que la logística casi nunca acertaba: su recall pasa de 0.06 a 0.68. El estrato 6 sube de 0.21 a 0.71. En cambio,
el estrato 5 empeora (de 0.19 a 0.04: el 77 % se predice como 4) y el 3 baja de 0.88 a 0.73. El Moran de los
residuos solo baja de 0.70 a 0.64.

En **validación cruzada**, el modelo principal mejora en 4 de los 5 folds, pero empeora 0.13 en el fold 1, y la
diferencia media (+0.025) no supera un error estándar. **Por el criterio fijado de antemano, no reemplaza a B.**

La única variante que cumple el criterio en CV es **A + vecindad, sin coordenadas** (+0.056, error estándar 0.028),
pero su intervalo de test incluye el cero. Es una variante exploratoria y se compararon cinco candidatos contra la
misma referencia, sin corrección por comparaciones múltiples. Aun así es un indicio interesante: la vecindad física
podría sustituir a la posición absoluta.

### Controles

**¿Hay fuga por usar las viviendas de test para calcular la vecindad?** No hay fuga de estrato, pero sí se usan los
atributos físicos de viviendas de test para describir el entorno de las de entrenamiento. En una **variante
estricta**, la vecindad se calcula solo dentro de cada partición. El F1 de test queda en 0.568 (Δ = +0.118, IC
[+0.011, +0.219]): la mejora se mantiene, aunque con menos evidencia, y el intervalo del MAE ya incluye el cero
([−0.428, +0.015]).

**¿Depende de un solo bloque?** La mejora se concentra en un bloque de 10 873 viviendas, cuya accuracy pasa de 0.19 a
0.76, y un bloque de 1 982 viviendas empeora (de 0.73 a 0.61). Al quitar un bloque a la vez, la mejora de F1 se
mantiene positiva en los siete casos (entre +0.095 y +0.210; con la variante estricta, entre +0.076 y +0.188).

**¿Se confirma con más folds?** Con 5 folds hay pocas comparaciones por pares, así que se repitió la validación con
**15 folds** dentro del entrenamiento (el test no interviene), con C = 0.01 sin pesos, la combinación elegida para
todos los modelos:

| Modelo | F1 agregado | Δ F1 medio por fold (error est.) | Folds en que mejora | Δ F1 agregado (IC 95 %) | Δ MAE (IC 95 %) |
|---|---|---|---|---|---|
| Logística B | 0.333 | — | — | — | — |
| B + vecindad | 0.397 | +0.006 (0.021) | 7 de 15 | +0.064 [−0.029, +0.160] | −0.064 [−0.285, +0.115] |
| A + vecindad (sin coordenadas) | 0.386 | +0.008 (0.023) | 7 de 15 | +0.054 [−0.040, +0.144] | +0.002 [−0.254, +0.270] |

**La CV más fina no confirma la mejora.** Ningún modelo cumple el criterio, todos los intervalos incluyen el cero y
la vecindad mejora en solo 7 de los 15 folds. Los dos modelos con vecindad mejoran y empeoran en los mismos folds
(por ejemplo, +0.23 en uno y entre −0.05 y −0.10 en otros cinco), lo que apunta a una propiedad de esas zonas y no a
ruido del modelo.

## 5.5 ¿Dónde ayuda y dónde perjudica la vecindad?

Este análisis es **exploratorio y posterior a los resultados**. Usa las predicciones fuera de muestra de la CV de 15
folds y las de test. Para diagnosticar, y solo para eso, se calcula el **estrato medio de las viviendas de otros
edificios a 300 m** y su dispersión. Esa información nunca entra a un modelo.

La hipótesis es que la vecindad **suaviza**: empuja la predicción hacia lo típico del entorno.

| Entorno de la vivienda (desviación del estrato a 300 m) | % de viviendas (CV / test) | Cambio en accuracy, CV | Cambio en accuracy, test |
|---|---|---|---|
| Muy homogéneo (< 0.3) | 27 % / 37 % | +0.10 | +0.09 |
| Moderado (0.3 a 1.0) | 71 % / 62 % | +0.03 a +0.10 | +0.06 a +0.19 |
| **Muy mezclado (> 1.0)** | 2 % / 0.5 % | **−0.16** | **−0.18** |

Es el patrón más consistente: se repite con la misma dirección en entrenamiento y en test. La vecindad ayuda a casi
todas las viviendas, pero perjudica mucho a las pocas que están en entornos muy mezclados (el MAE sube entre 0.5 y
0.8). Las viviendas con un estrato **más alto** que el de su entorno también tienden a empeorar en la CV, de forma
coherente con la idea de suavizado.

Lo que **no** se sostiene:

- **Los efectos por estrato cambian de signo entre CV y test.** Por ejemplo, el estrato 5 empeora en test (−0.15 de
  recall) y mejora en la CV (+0.30). No se pueden sacar conclusiones por estrato.
- **Por bloques no aparece una característica clara.** Entre 25 bloques con al menos 1 000 viviendas, hay solo una
  tendencia débil a que ayude más en los bloques del norte y de estrato alto (Spearman ≈ −0.35, p ≈ 0.06 a 0.09).
- **Hipótesis descartada: vivienda social en apartamentos.** Se planteó que la vecindad fallaría en zonas de
  apartamentos de estrato bajo (VIS/VIP), donde "muchos apartamentos alrededor" no significa estrato alto como en el
  norte. Explica un solo bloque del sur: de los cuatro bloques con ese perfil, solo uno empeora, y la correlación con
  el porcentaje de VIS/VIP del bloque es nula (ρ = +0.03).

**Implicación:** la vecindad física aporta información útil, pero el modelo no sabe cuándo no confiar en ella. Un
paso natural es agregar medidas de **qué tan variado es físicamente el entorno** (por ejemplo, la dispersión del área
o de los baños a 300 m), que no usan el estrato y le permitirían restarle peso a la vecindad en barrios mezclados.

## 5.6 Escenario 2: zonas conocidas

El capítulo 4 concluyó que el XGBoost geográfico no aporta en zonas nuevas porque sus modelos locales no tienen
datos ahí, y dejó pendiente medirlo **dentro de zonas conocidas**. Así se hizo:

- **Datos:** las 218 870 viviendas de los bloques de entrenamiento. El test del Entregable 1 no interviene.
- **Partición:** se ocultan **edificios completos** al azar en 5 folds (semilla 42), sin buffer. El modelo ve a los
  vecinos de la vivienda, pero nunca otras unidades de su mismo edificio. Ocultar filas sueltas sería fuga, porque el
  estrato casi no varía dentro de un edificio o conjunto (ICC = 0.992).
- **Información legítima en este escenario:** el estrato de las viviendas conocidas cercanas. En una zona ya
  estratificada esa información existe, así que aquí se usa. En zonas nuevas sería fuga.
- **Hiperparámetros:** los elegidos por la validación espacial, sin una búsqueda nueva. El XGBoost geográfico usa la
  ventana del 15 % y $\alpha$ = 0.75, con 33 o 34 anclas por fold, que cubren el 100 % de las viviendas ocultas.

| Modelo | F1 macro | Accuracy | MAE ordinal | Kappa cuadrático |
|---|---|---|---|---|
| Vecinos más cercanos (voto de las 15 viviendas conocidas) | **0.788** | **0.807** | **0.226** | **0.929** |
| XGBoost geográfico | 0.784 | 0.803 | 0.236 | 0.925 |
| XGBoost | 0.774 | 0.795 | 0.244 | 0.923 |
| Logística B + vecindad + estrato de los vecinos | 0.718 | 0.743 | 0.299 | 0.907 |
| Logística B + vecindad | 0.685 | 0.712 | 0.324 | 0.903 |
| Logística B | 0.574 | 0.639 | 0.436 | 0.853 |
| Moda por zona (2 km) | 0.531 | 0.652 | 0.420 | 0.845 |

Diferencias pareadas, con IC 95 % por bootstrap de los 30 bloques:

| Comparación | Δ F1 macro | Δ MAE ordinal |
|---|---|---|
| **XGBoost geográfico − XGBoost** | **+0.010 [+0.007, +0.024]** | **−0.009 [−0.013, −0.005]** |
| XGBoost − logística B | +0.200 [+0.135, +0.244] | −0.191 [−0.253, −0.132] |
| Logística B + vecindad − logística B | +0.111 [+0.046, +0.142] | −0.112 [−0.164, −0.065] |
| Logística B + vecindad + estrato de vecinos − logística B | +0.144 [+0.072, +0.174] | −0.137 [−0.193, −0.068] |
| Logística B − moda por zona | +0.044 [−0.051, +0.072] | +0.016 [−0.036, +0.067] |
| XGBoost geográfico − vecinos más cercanos | −0.004 [−0.039, +0.015] | +0.009 [−0.018, +0.036] |
| Logística B + vecindad + estrato de vecinos − vecinos más cercanos | −0.070 [−0.103, −0.052] | +0.073 [+0.039, +0.110] |

Cuatro lecturas:

1. **El mejor modelo depende del escenario.** En zonas nuevas, XGBoost perdía frente a la logística (F1 de test 0.30
   frente a 0.45). En zonas conocidas la supera por +0.20. Los árboles aprenden bien la geografía donde hay datos y
   no saben extrapolarla a zonas sin datos. Por eso hay que reportar cada escenario por separado.
2. **El XGBoost geográfico sí aporta en su escenario natural.** Supera a su control en los 5 folds (de +0.005 a
   +0.022) y los intervalos de F1 y de MAE excluyen el cero. La mejora es pequeña (+0.010), pero consistente, a
   diferencia de lo que pasó en zonas nuevas (+0.003, sin importancia práctica).
3. **Pero empata con la regla más simple.** Votar con el estrato de las 15 viviendas conocidas más cercanas da F1
   0.788, frente a 0.784 del XGBoost geográfico, y la diferencia no es distinguible de cero. Donde ya se conocen los
   estratos del barrio, las variables catastrales añaden poco a la ubicación.
4. **La vecindad física también ayuda aquí** (+0.111 sobre la logística B). Pero incluso con el estrato de los
   vecinos, la logística (0.718) queda muy por debajo del voto de los vecinos (0.788): un modelo lineal no aprovecha
   bien la información local.

## 5.7 Conclusión y limitaciones

**¿Son mejores los modelos nuevos? Depende del escenario, y esa es la conclusión principal del capítulo.**

- **En zonas nuevas,** que es el escenario de este informe, la logística B sigue siendo la referencia. Las
  variables de vecindad física mejoran el test, pero la validación cruzada no lo confirma. Su efecto depende de la
  zona: ayuda en barrios homogéneos y perjudica en los muy mezclados. Es una línea prometedora, no una mejora
  demostrada.
- **En zonas conocidas,** los árboles son claramente mejores que la logística, y el XGBoost geográfico, la propuesta
  novedosa del proyecto, supera de forma consistente a su control. Pero empata con copiar el estrato de los vecinos.

**Lectura a la luz de la metodología del DANE.** Los dos escenarios se explican por cómo se asigna el estrato (sección 1.1.1): el DANE califica primero la zona y le da su estrato a todas sus viviendas, y solo corrige las viviendas atípicas. En zonas conocidas, los vecinos ya revelan la zona, y por eso votar con su estrato es tan difícil de superar: lo que aprende el modelo es, en esencia, la zona del DANE. En zonas nuevas, la zona hay que deducirla de las viviendas, y el insumo que el DANE usa para eso (el puntaje de calificación de las edificaciones) no está en los datos abiertos. Las variables de vecindad física son un sustituto parcial de ese puntaje medio de la zona; que funcionen en barrios homogéneos y fallen en los mezclados es coherente con una metodología pensada para zonas homogéneas.

**Implicación práctica.** Para estimar el estrato de un predio en una zona ya estratificada, basta una regla de
vecinos. Los modelos con variables catastrales solo son necesarios donde no hay estratos conocidos alrededor, y ahí
el problema sigue abierto. Los siguientes pasos son medir la variedad física del entorno, para que el modelo sepa
cuándo confiar en la vecindad, y evaluar con más bloques para poder distinguir mejoras pequeñas.

**Limitaciones de este capítulo:**

- **Réplica sin scikit-learn.** Reproduce el modelo B al milésimo, pero no es el código original.
- **Pocos bloques de test.** Los 7 bloques de test hacen que los intervalos sean frágiles. La CV de 15 folds da más
  comparaciones, pero sigue usando solo 30 bloques.
- **Comparaciones múltiples.** Se evaluaron cinco variantes contra la misma referencia, sin corrección. Por eso solo
  el modelo principal, fijado de antemano, se interpreta como prueba.
- **Análisis por zonas posterior.** La sección 5.5 es exploratoria y sus explicaciones son hipótesis.
- **Hiperparámetros del escenario 2.** En zonas conocidas no se volvieron a buscar: se usaron los elegidos por la
  validación espacial.
