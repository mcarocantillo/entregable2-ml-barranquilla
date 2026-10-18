# 5. Variables de vecindad física y los dos escenarios de predicción

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

Este capítulo corresponde a las diapositivas 5.7 a 5.11 de la pestaña **5. Más allá del modelo base** del dashboard.

```{admonition} Revisado en la versión 3
:class: important
Las cifras de este capítulo usan la validación con buffer de 1 km. El capítulo 6 muestra que ese buffer evaluaba a 1.4–1.5 km del dato más cercano, más lejos que el escenario declarado (0.45 km). Con una validación que reproduce esa distancia (kNNDM), la vecindad física **sí** mejora las zonas nuevas de forma significativa (F1 de test de 0.49 a 0.57–0.60), lo que corrige el «resultado mixto» de este capítulo.
```

Los capítulos 3 y 4 dejaron dos pistas. Los residuos de la logística siguen agrupados en el espacio (Moran = 0.56),
así que al modelo le falta información de **barrio**. Y los modelos más flexibles no ayudan en **zonas nuevas**, lo
que sugiere que el límite está en la información disponible y no en el algoritmo. Este capítulo pone a prueba esas
dos ideas con dos experimentos:

1. **Variables de vecindad física** en el escenario de siempre (zonas nuevas): ¿mejora la logística si, además de la
   vivienda, ve cómo son las viviendas que la rodean?
2. **El escenario de zonas conocidas**, que el capítulo 4 dejó pendiente: predecir viviendas ocultas dentro de zonas
   con datos, donde el XGBoost geográfico debería rendir mejor.

```{admonition} Resumen del capítulo
:class: important
- **Zonas nuevas.** Las variables de vecindad física mejoran las métricas ordinales del test (MAE −0.11) y suben el
  F1 de 0.49 a 0.55, pero esa mejora de F1 no es significativa y la validación cruzada (con 5 y con 15 folds) no la
  confirma. Es un **resultado mixto**: la logística B sigue siendo la referencia.
- **Zonas conocidas.** El orden se invierte: los árboles superan a la logística por +0.23 de F1 ocultando edificios y
  por +0.20 ocultando **manzanas completas**. Ocultando edificios, el voto de las 15 viviendas conocidas más cercanas
  empata con el XGBoost geográfico (0.79); ocultando manzanas, el voto cae a 0.71 y el XGBoost geográfico lo supera
  (0.75; IC de la diferencia [+0.001, +0.070]).
```

## 5.1 Cómo se hizo: una réplica verificada del protocolo

El código está en la carpeta `vecindad/` del repositorio (ver su `README.md`). En el equipo donde se corrió, una
política de control de aplicaciones de Windows bloquea las librerías compiladas de scikit-learn. Por eso el
preprocesamiento y la regresión logística se **reimplementaron con numpy y scipy**, con las mismas fórmulas del
Entregable 1. La limpieza, la partición y los folds se tomaron sin cambios del código del Entregable 1 (versión 2, con
la recuperación por manzana).

Antes de usar la réplica se comprobó que reproduce las cifras guardadas del modelo B:

| | Réplica | Entregable 1 |
|---|---|---|
| Viviendas de entrenamiento / tras el buffer / de test | 257 273 / 144 586 / 65 145 | 257 273 / 144 586 / 65 145 |
| F1 macro por fold (CV espacial) | 0.169, 0.375, 0.357, 0.409, 0.300 | 0.168, 0.375, 0.357, 0.409, 0.300 |
| F1 macro en CV (media) | 0.322 | 0.322 |
| Combinación elegida por la regla 1-SE (B / A) | C = 0.01 balanceado / C = 0.01 sin pesos | igual |
| F1 macro en test | 0.4919 | 0.4921 |
| Accuracy / MAE ordinal / kappa en test | 0.611 / 0.439 / 0.835 | 0.611 / 0.439 / 0.835 |
| Moran de los residuos | 0.559 | 0.559 |

Las diferencias son de menos de una milésima, así que las comparaciones de este capítulo usan el mismo protocolo que
el modelo base.

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
las viviendas probadas. La mediana de viviendas vecinas es de 1 194 a 300 m y de 11 850 a 1 km, y solo el 0.4 % de
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
| Logística B (referencia) | 0.322 | — | — | 0.492 | — |
| **B + vecindad (300 m y 1 km)** | 0.359 | +0.038 (0.045) | No | **0.554** | +0.062 [−0.027, +0.108] |
| B + vecindad, solo 300 m | 0.390 | +0.068 (0.027) | Sí | 0.551 | +0.059 [−0.035, +0.112] |
| B + vecindad, solo 1 km | 0.386 | +0.064 (0.042) | Sí | 0.559 | +0.067 [−0.021, +0.108] |
| A: solo variables físicas | 0.286 | −0.036 (0.041) | No | 0.370 | −0.122 [−0.146, −0.034] |
| A + vecindad (sin coordenadas) | 0.384 | +0.062 (0.034) | Sí | 0.527 | +0.035 [−0.025, +0.098] |

En **test**, el modelo principal sube el F1 de 0.492 a 0.554, pero el intervalo de la diferencia incluye el cero. Las
métricas ordinales sí mejoran de forma significativa: el MAE ordinal baja de 0.439 a 0.334 (IC de la diferencia
[−0.223, −0.031]) y el kappa cuadrático sube de 0.835 a 0.896 (IC [+0.036, +0.162]); la accuracy pasa de 0.611 a
0.685. El cambio más visible está en los estratos **4 y 6**, que la logística acertaba poco: el recall del 4 pasa de
0.28 a 0.65 y el del 6 de 0.21 a 0.70; el estrato 1 sube de 0.79 a 0.90. En cambio, el **estrato 5 se desploma** (de
0.42 a 0.12) y el 2 y el 3 bajan un poco (de 0.60 a 0.53 y de 0.70 a 0.64). El Moran de los residuos no baja: pasa
de 0.56 a 0.61.

En **validación cruzada**, el modelo principal mejora en 3 de los 5 folds (+0.19 en el fold 1) y empeora en dos
(−0.08 en el fold 4), y la diferencia media (+0.038) no supera un error estándar (0.045). **Por el criterio fijado de
antemano, no reemplaza a B.**

Tres variantes exploratorias sí cumplen el criterio en CV: **una sola escala** (300 m o 1 km) y **A + vecindad, sin
coordenadas** (+0.062, error estándar 0.034). Ninguna tiene un intervalo de test que excluya el cero, y se compararon
cinco candidatos contra la misma referencia sin corrección por comparaciones múltiples. Aun así hay dos indicios
interesantes: con una sola escala la mejora es más estable entre folds que con las dos juntas (las 12 variables son
muy redundantes entre sí), y la vecindad física podría sustituir a la posición absoluta.

### Controles

**¿Hay fuga por usar las viviendas de test para calcular la vecindad?** No hay fuga de estrato, pero sí se usan los
atributos físicos de viviendas de test para describir el entorno de las de entrenamiento. En una **variante
estricta**, la vecindad se calcula solo dentro de cada partición (con los hiperparámetros de B, C = 0.01 balanceado,
en ambas versiones). El F1 de test queda en 0.553, igual que con la vecindad calculada con todas las viviendas
(0.551): la mejora no viene de esa información (Δ frente a B = +0.061, IC [−0.032, +0.088]; MAE −0.079, IC [−0.204,
−0.005]).

**¿Depende de un solo bloque?** Sí, en buena parte. La mejora se concentra en el bloque de test más grande del norte
(15 547 viviendas), cuya accuracy pasa de 0.35 a 0.46 (0.57 con la variante estricta). Al quitar ese bloque, la mejora
de F1 baja de +0.06 a +0.004; al quitar cualquiera de los otros seis, se mantiene entre +0.059 y +0.067.

**¿Se confirma con más folds?** Con 5 folds hay pocas comparaciones por pares, así que se repitió la validación con
**15 folds** dentro del entrenamiento (el test no interviene), con C = 0.01 y pesos balanceados, la combinación
elegida para B:

| Modelo | F1 agregado | Δ F1 medio por fold (error est.) | Folds en que mejora | Δ F1 agregado (IC 95 %) | Δ MAE (IC 95 %) |
|---|---|---|---|---|---|
| Logística B | 0.386 | — | — | — | — |
| B + vecindad | 0.415 | +0.023 (0.025) | 10 de 15 | +0.029 [−0.063, +0.094] | −0.113 [−0.311, +0.100] |
| A + vecindad (sin coordenadas) | 0.418 | +0.014 (0.023) | 9 de 15 | +0.032 [−0.038, +0.085] | −0.077 [−0.254, +0.168] |

**La CV más fina no confirma la mejora.** Ningún modelo cumple el criterio y todos los intervalos incluyen el cero.
Los dos modelos con vecindad mejoran y empeoran en los mismos folds (por ejemplo, +0.16 y +0.10 en el fold 1, −0.16
y −0.17 en el fold 5), lo que apunta a una propiedad de esas zonas y no a ruido del modelo.

## 5.5 ¿Dónde ayuda y dónde perjudica la vecindad?

Este análisis es **exploratorio y posterior a los resultados**. Usa las predicciones fuera de muestra de la CV de 15
folds y las de test. Para diagnosticar, y solo para eso, se calcula el **estrato medio de las viviendas de otros
edificios a 300 m** y su dispersión. Esa información nunca entra a un modelo.

La hipótesis era que la vecindad **suaviza**: empuja la predicción hacia lo típico del entorno.

| Entorno de la vivienda (desviación del estrato a 300 m) | % de viviendas (CV / test) | Cambio en accuracy, CV | Cambio en accuracy, test |
|---|---|---|---|
| Muy homogéneo (< 0.3) | 30 % / 27 % | −0.02 | +0.00 |
| 0.3 a 0.6 | 51 % / 55 % | +0.03 | +0.14 |
| 0.6 a 1.0 | 17 % / 17 % | +0.24 | −0.01 |
| **Muy mezclado (> 1.0)** | 2 % / 0.5 % | **+0.16** | **−0.23** |

En test el patrón es el del suavizado: la vecindad ayuda a las viviendas parecidas a su entorno (75 % del test, +0.14
de accuracy) y perjudica a las atípicas y a las de entornos muy mezclados. Pero **en la CV no se repite**: ahí los
entornos mezclados mejoran, y las viviendas de estrato mucho más bajo que su entorno pasan de 0.02 a 0.77 de
accuracy. Con los datos de la versión 2, la hipótesis del suavizado no se sostiene como regla general.

Lo que **sí** aparece de forma clara es la **geografía**: entre 29 bloques con al menos 1 000 viviendas, la vecindad
ayuda más cuanto más al norte está el bloque (Spearman entre el cambio de MAE y la coordenada norte-sur ρ = −0.62,
p < 0.001). Las demás características del bloque tienen asociaciones débiles (estrato medio ρ = −0.31, p = 0.11;
dispersión del estrato ρ = −0.33, p = 0.09).

Lo que **no** se sostiene:

- **Los efectos por estrato cambian de signo entre CV y test.** El estrato 4 mejora en test (+0.37 de recall) y
  empeora en la CV (−0.19), y con el 5 y el 6 pasa lo mismo en sentido contrario. No se pueden sacar conclusiones por
  estrato.
- **Hipótesis de la vivienda social en apartamentos.** Se planteó que la vecindad fallaría en zonas de apartamentos
  de estrato bajo (VIS/VIP), donde "muchos apartamentos alrededor" no significa estrato alto como en el norte. De los
  cuatro bloques con ese perfil, solo uno empeora, y la correlación con el porcentaje de VIS/VIP del bloque es nula
  (ρ = +0.06). El caso más extremo de toda la tabla sí va en esa dirección (un bloque de 1 198 apartamentos de estrato
  1 cuya accuracy cae de 1.00 a 0.05), pero es un solo bloque.

**Implicación:** la vecindad física aporta información útil, sobre todo en el norte, pero el modelo no sabe cuándo no
confiar en ella. Un paso natural es agregar medidas de **qué tan variado es físicamente el entorno** (por ejemplo, la
dispersión del área o de los baños a 300 m), que no usan el estrato y le permitirían restarle peso a la vecindad en
barrios mezclados.

## 5.6 Escenario 2: zonas conocidas

El capítulo 4 concluyó que el XGBoost geográfico no aporta de forma estable en zonas nuevas porque sus modelos locales
no tienen datos ahí, y dejó pendiente medirlo **dentro de zonas conocidas**. Así se hizo:

- **Datos:** las 257 273 viviendas de los bloques de entrenamiento. El test del Entregable 1 no interviene.
- **Partición:** se ocultan **edificios completos** al azar en 5 folds (semilla 42), sin buffer. El modelo ve a los
  vecinos de la vivienda, pero nunca otras unidades de su mismo edificio. Ocultar filas sueltas sería fuga, porque el
  estrato casi no varía dentro de un edificio o conjunto (ICC = 0.992).
- **Información legítima en este escenario:** el estrato de las viviendas conocidas cercanas. En una zona ya
  estratificada esa información existe, así que aquí se usa. En zonas nuevas sería fuga.
- **Hiperparámetros:** los elegidos por la validación espacial, sin una búsqueda nueva. El XGBoost geográfico usa la
  ventana del 15 % y $\alpha$ = 0.75, con 34 o 35 anclas por fold, que cubren el 100 % de las viviendas ocultas.

| Modelo | F1 macro | Accuracy | MAE ordinal | Kappa cuadrático |
|---|---|---|---|---|
| Vecinos más cercanos (voto de las 15 viviendas conocidas) | **0.794** | **0.822** | **0.206** | **0.935** |
| XGBoost geográfico | 0.793 | 0.819 | 0.213 | 0.932 |
| XGBoost | 0.787 | 0.812 | 0.220 | 0.931 |
| Logística B + vecindad + estrato de los vecinos | 0.741 | 0.756 | 0.268 | 0.923 |
| Logística B + vecindad | 0.701 | 0.717 | 0.313 | 0.910 |
| Logística B | 0.560 | 0.602 | 0.483 | 0.845 |
| Moda por zona (2 km) | 0.529 | 0.656 | 0.398 | 0.859 |

Diferencias pareadas, con IC 95 % por bootstrap de los 32 bloques:

| Comparación | Δ F1 macro | Δ MAE ordinal |
|---|---|---|
| **XGBoost geográfico − XGBoost** | **+0.006 [+0.003, +0.008]** | **−0.006 [−0.011, −0.003]** |
| XGBoost − logística B | +0.227 [+0.156, +0.291] | −0.263 [−0.345, −0.194] |
| Logística B + vecindad − logística B | +0.141 [+0.096, +0.187] | −0.170 [−0.234, −0.111] |
| Logística B + vecindad + estrato de vecinos − logística B | +0.181 [+0.146, +0.241] | −0.215 [−0.289, −0.143] |
| Logística B − moda por zona | +0.031 [−0.081, +0.063] | +0.085 [+0.024, +0.151] |
| XGBoost geográfico − vecinos más cercanos | −0.001 [−0.071, +0.017] | +0.008 [−0.018, +0.030] |
| Logística B + vecindad + estrato de vecinos − vecinos más cercanos | −0.053 [−0.072, −0.026] | +0.063 [+0.038, +0.092] |

Cuatro lecturas:

1. **El mejor modelo depende del escenario.** En zonas nuevas, XGBoost perdía frente a la logística (F1 de test 0.375
   frente a 0.492). En zonas conocidas la supera por +0.23. Los árboles aprenden bien la geografía donde hay datos y
   no saben extrapolarla a zonas sin datos. Por eso hay que reportar cada escenario por separado.
2. **El XGBoost geográfico aporta poco, pero de forma consistente, en su escenario natural.** Supera a su control en
   los 5 folds (entre +0.003 y +0.007) y los intervalos de F1 y de MAE excluyen el cero. La mejora es muy pequeña
   (+0.006).
3. **Pero, ocultando solo edificios, empata con la regla más simple** (ver 5.7). Votar con el estrato de las 15 viviendas conocidas más cercanas da F1
   0.794, frente a 0.793 del XGBoost geográfico, y la diferencia no es distinguible de cero. Donde ya se conocen los
   estratos del barrio, las variables catastrales añaden poco a la ubicación: XGBoost no es "más inteligente" que la
   logística, sino que **memoriza el mapa** con las coordenadas, algo que un modelo lineal no puede hacer.
4. **La vecindad física también ayuda aquí** (+0.141 sobre la logística B). Pero incluso con el estrato de los
   vecinos, la logística (0.741) queda por debajo del voto de los vecinos (0.794): un modelo lineal no aprovecha bien
   la información local.

## 5.7 Control: ¿la ventaja es memorizar la manzana?

Ocultar edificios evita que el modelo vea otras unidades del mismo edificio, pero deja en el entrenamiento a sus
**vecinos de manzana**. Eso puede inflar este escenario por dos razones: en Colombia el estrato se asigna por manzana
(o lado de manzana), y en la versión 2 las viviendas ubicadas por el centroide de su manzana comparten exactamente la
misma coordenada. Para medirlo se repitió el escenario ocultando **manzanas completas** (los 17 primeros dígitos del
número predial; la manzana comodín «0000» se agrupa por edificio): 6 634 grupos en 5 folds, con la misma semilla y los
mismos hiperparámetros (`AGRUPAR=manzana` en `vecindad/06_zonas_conocidas.py`).

| Modelo | F1 ocultando edificios | F1 ocultando manzanas | Cambio |
|---|---|---|---|
| Vecinos más cercanos (voto de las 15 viviendas conocidas) | 0.794 | 0.710 | **−0.084** |
| XGBoost geográfico | 0.793 | **0.754** | −0.039 |
| XGBoost | 0.787 | 0.748 | −0.039 |
| Logística B + vecindad + estrato de los vecinos | 0.741 | 0.701 | −0.040 |
| Logística B + vecindad | 0.701 | 0.673 | −0.028 |
| Logística B | 0.560 | 0.547 | −0.013 |
| Moda por zona (2 km) | 0.529 | 0.517 | −0.012 |

Diferencias pareadas ocultando manzanas, con IC 95 % por bootstrap de los 32 bloques:

| Comparación | Δ F1 macro | Δ MAE ordinal |
|---|---|---|
| XGBoost − logística B | +0.201 [+0.156, +0.269] | −0.250 [−0.327, −0.188] |
| XGBoost geográfico − XGBoost | +0.006 [+0.002, +0.015] | −0.007 [−0.010, −0.004] |
| **XGBoost geográfico − vecinos más cercanos** | **+0.044 [+0.001, +0.070]** | **−0.052 [−0.078, −0.028]** |
| Logística B + vecindad + estrato de vecinos − vecinos más cercanos | −0.010 [−0.061, +0.020] | +0.021 [−0.015, +0.068] |

Tres lecturas:

1. **Sí había una parte de «copiar la manzana».** El voto de los vecinos, que solo copia estratos, es el que más
   pierde (−0.08). Los modelos con variables pierden la mitad (−0.04), y la logística casi nada (−0.01), porque no
   puede memorizar el mapa.
2. **La ventaja de los árboles no era fuga.** Con manzanas completas ocultas, XGBoost sigue superando a la logística
   por +0.20, con un intervalo lejos del cero.
3. **Con la prueba más estricta, la propuesta novedosa sí aporta frente a la regla simple.** El XGBoost geográfico
   supera al voto de los vecinos (+0.044 de F1 y −0.05 de MAE, ambos con intervalos que excluyen el cero) y sigue
   mejorando a su control de forma pequeña pero consistente.

## 5.8 Conclusión y limitaciones

**¿Son mejores los modelos nuevos? Depende del escenario, y esa es la conclusión principal del capítulo.**

- **En zonas nuevas,** que es el escenario de este informe, la logística B sigue siendo la referencia. Las variables
  de vecindad física mejoran las métricas ordinales del test, pero la mejora de F1 no es significativa, la validación
  cruzada no la confirma y depende sobre todo de un bloque del norte. Es una línea prometedora, no una mejora
  demostrada.
- **En zonas conocidas,** los árboles son claramente mejores que la logística, también cuando se ocultan manzanas
  completas (+0.20 de F1). Ocultando solo edificios, el XGBoost geográfico, la propuesta novedosa del proyecto,
  empata con copiar el estrato de los vecinos; ocultando manzanas, lo supera. Frente a su control, aporta poco pero
  de forma consistente en las dos pruebas.

**Lectura a la luz de la metodología del DANE.** Los dos escenarios se explican por cómo se asigna el estrato (sección 1.1.1): el DANE califica primero la zona y le da su estrato a todas sus viviendas, y solo corrige las viviendas atípicas. En zonas conocidas, los vecinos ya revelan la zona, y por eso votar con su estrato es tan difícil de superar cuando se conocen viviendas de la misma manzana: lo que aprende el modelo es, en esencia, la zona del DANE. Cuando la manzana entera está oculta, la zona hay que deducirla de las manzanas vecinas, y ahí los árboles con las variables catastrales le ganan a la copia. En zonas nuevas, la zona hay que deducirla de las viviendas, y el insumo que el DANE usa para eso (el puntaje de calificación de las edificaciones) no está en los datos abiertos. Las variables de vecindad física son un sustituto parcial de ese puntaje medio de la zona.

**Implicación práctica.** Para estimar el estrato de un predio en una zona ya estratificada, basta una regla de
vecinos. Los modelos con variables catastrales solo son necesarios donde no hay estratos conocidos alrededor, y ahí
el problema sigue abierto. Los siguientes pasos son medir la variedad física del entorno, para que el modelo sepa
cuándo confiar en la vecindad, y evaluar con más bloques para poder distinguir mejoras pequeñas.

**Limitaciones de este capítulo:**

- **Réplica sin scikit-learn.** Reproduce el modelo B al milésimo, pero no es el código original.
- **Pocos bloques de test.** Los 7 bloques de test hacen que los intervalos sean frágiles. La CV de 15 folds da más
  comparaciones, pero sigue usando solo 32 bloques.
- **Comparaciones múltiples.** Se evaluaron cinco variantes contra la misma referencia, sin corrección. Por eso solo
  el modelo principal, fijado de antemano, se interpreta como prueba.
- **Análisis por zonas posterior.** La sección 5.5 es exploratoria y sus explicaciones son hipótesis.
- **Hiperparámetros del escenario 2.** En zonas conocidas no se volvieron a buscar: se usaron los elegidos por la
  validación espacial.
