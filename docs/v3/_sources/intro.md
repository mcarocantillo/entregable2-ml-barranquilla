# Estrato socioeconómico y catastro en Barranquilla — Entregable 2

**Machine Learning — Maestría, Universidad del Norte · Profesor Lihki Rubio**

**Autores:** María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Octubre de 2026

## Qué contiene este informe

Este libro acompaña al **dashboard en Dash Plotly** del proyecto, que reúne los Entregables 1 y 2. El dashboard tiene seis pestañas, cada una como una secuencia de diapositivas interactivas: **Contexto**, **Datos y calidad**, **Análisis exploratorio**, **Modelo base**, **Más allá del modelo base** (modelos de la revisión, vecindad y escenarios) y **Conclusiones**, además de un glosario que explica qué mide cada métrica y cómo leer su valor. Aquí se interpreta cada resultado que aparece en él y se agregan los detalles que no caben en un tablero: cómo se calculó cada cifra, qué decisiones se tomaron y por qué, y las limitaciones.

El dashboard está desplegado en Render: **https://entregable2-ml-barranquilla.onrender.com**. El servicio gratuito se suspende cuando no tiene visitas, así que la primera carga puede tardar alrededor de un minuto.

## El problema en una frase

¿Se puede predecir el **estrato socioeconómico** (1 a 6) de una vivienda de Barranquilla a partir de sus **características catastrales** (área, baños, habitaciones, pisos, antigüedad, régimen de propiedad) y de su **ubicación**?

## Resumen de resultados

Se trabajó con **332 718 viviendas** del catastro abierto (168 044 predios matriz: edificios, conjuntos o lotes). Con la versión 2 del Entregable 1, las viviendas sin coordenadas propias se ubican con el centroide de su manzana, y el modelo cubre el **96.9 %** de ellas. El estrato está desbalanceado (6.2 a 1 entre la clase más y la menos frecuente del entrenamiento), es ordinal y tiene una autocorrelación espacial muy fuerte (I de Moran = 0.92): la ciudad está segregada de sur a norte. Por eso la partición entrenamiento/prueba y la validación cruzada se hicieron por **bloques espaciales de 2 km con un buffer de 1 km**, y la incertidumbre se midió remuestreando bloques.

El modelo base, una **regresión logística multinomial** con variables físicas y de ubicación y pesos de clase balanceados, obtiene un **F1 macro de 0.32 ± 0.08 en validación cruzada espacial y de 0.49 en test** (IC 95 % [0.28, 0.51]). Supera con claridad a las líneas base Dummy. También supera en todas las métricas a la línea base geográfica, la clase más frecuente por zona de 4 km, aunque esa diferencia de F1 no es significativa. Los errores son casi siempre entre estratos vecinos (accuracy ±1 = 0.96). Los residuos siguen autocorrelacionados en el espacio (Moran = 0.56).

Además de lo que pide el enunciado, se evaluaron con el mismo protocolo los candidatos de la revisión bibliográfica: una **adaptación propia del XGBoost geográfico a clasificación** (la propuesta novedosa), XGBoost, Rotation Forest y una SVM con kernel RBF optimizada con TPE. Ninguno supera a la logística en test. La SVM pasa el criterio de la validación cruzada, pero en test queda por debajo (F1 0.45 frente a 0.49) y con errores más grandes, así que se mantiene la logística como desviación declarada del criterio; Rotation Forest empata en F1 con un error ordinal menor; el XGBoost geográfico no aporta de forma estable porque sus modelos locales no tienen datos en las zonas que evalúa la validación espacial. El capítulo 5 mide por separado los dos escenarios de predicción. En **zonas conocidas** el orden se invierte: los árboles superan a la logística por más de 0.2 de F1, incluso ocultando manzanas completas, y en esa prueba estricta el XGBoost geográfico supera a votar con el estrato de las viviendas vecinas. En **zonas nuevas**, agregar variables de vecindad física da un resultado mixto: mejora las métricas ordinales del test, pero la validación cruzada no lo confirma. La **versión 3** (capítulo 6), a partir de la sugerencia del profesor de usar herramientas de vecinos más cercanos, reemplaza el buffer fijo por una validación que empareja distancias al vecino más cercano (kNNDM): con ella la validación cruzada anticipa el test y la vecindad física sí mejora las zonas nuevas (F1 de test de 0.49 a 0.57–0.60, significativo); en zonas conocidas, la logística con el estrato de los vecinos alcanza a los árboles (0.76).

## Relación con el Entregable 1

El EDA completo, la limpieza y el diseño de la validación se documentaron en el Entregable 1. Para el modelo base, el dashboard **no reentrena ni rehace decisiones**: usa la misma partición, los mismos hiperparámetros y las mismas semillas. Los modelos avanzados del capítulo 4 sí se ajustan en este entregable, con la misma partición y los mismos folds. El script `preparar_datos.py` reproduce las métricas del Entregable 1 (versión 2) al tercer decimal (F1 de test 0.492 para el modelo principal) antes de guardar los resultados que muestra el tablero.

```{note}
**Cómo leer este libro.** El capítulo 1 corresponde a las pestañas Contexto y Datos y calidad, el 2 a Análisis exploratorio, el 3 a Modelo base y los capítulos 4 y 5 a Más allá del modelo base. Las figuras son las mismas, generadas por el mismo módulo (`figuras.py`), así que lo que se ve en el tablero y lo que se interpreta aquí coincide exactamente. El código de cada figura está oculto; se puede desplegar con el botón de cada celda.
```
