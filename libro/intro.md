# Estrato socioeconómico y catastro en Barranquilla — Entregable 2

**Machine Learning — Maestría, Universidad del Norte · Profesor Lihki Rubio**

**Autores:** María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Octubre de 2026

## Qué contiene este informe

Este libro acompaña al **dashboard en Dash Plotly** del Entregable 2. El dashboard tiene tres pestañas: **Contexto** (el problema, por qué importa y cómo se trabajó), **EDA** (el análisis exploratorio) y **ML models** (protocolo, modelo base, diagnóstico, modelos de la revisión bibliográfica, variables de vecindad y escenarios, simulador y conclusiones). Aquí se interpreta cada resultado que aparece en él y se agregan los detalles que no caben en un tablero: cómo se calculó cada cifra, qué decisiones se tomaron y por qué, y las limitaciones.

El dashboard está desplegado en Render: **https://entregable2-ml-barranquilla.onrender.com**. El servicio gratuito se suspende cuando no tiene visitas, así que la primera carga puede tardar alrededor de un minuto.

## El problema en una frase

¿Se puede predecir el **estrato socioeconómico** (1 a 6) de una vivienda de Barranquilla a partir de sus **características catastrales** (área, baños, habitaciones, pisos, antigüedad, régimen de propiedad) y de su **ubicación**?

## Resumen de resultados

Se trabajó con **332 718 viviendas** del catastro abierto (168 044 edificios). El estrato está moderadamente desbalanceado, es ordinal y tiene una autocorrelación espacial muy fuerte (I de Moran = 0.91): la ciudad está segregada de sur a norte. Por eso la partición entrenamiento/prueba y la validación cruzada se hicieron por **bloques espaciales de 2 km con un buffer de 1 km**, y la incertidumbre se midió remuestreando bloques.

El modelo base, una **regresión logística multinomial** con variables físicas y de ubicación, obtiene un **F1 macro de 0.30 ± 0.09 en validación cruzada espacial y de 0.45 en test** (IC 95 % [0.16, 0.46]). Supera con claridad a las líneas base Dummy. También supera en todas las métricas a la línea base geográfica, la clase más frecuente por zona de 2 km, aunque esa diferencia de F1 no es significativa. Los errores son casi siempre entre estratos vecinos (accuracy ±1 = 0.94). Los residuos siguen autocorrelacionados en el espacio (Moran = 0.70).

Además de lo que pide el enunciado, se evaluaron con el mismo protocolo los candidatos de la revisión bibliográfica: una **adaptación propia del XGBoost geográfico a clasificación** (la propuesta novedosa), XGBoost, Rotation Forest y una SVM con kernel RBF optimizada con TPE. Ninguno supera a la logística en zonas nuevas. La SVM queda cerca (F1 de test 0.43 frente a 0.45) y reconoce mejor los estratos altos; el XGBoost geográfico no aporta porque sus modelos locales no tienen datos en las zonas que evalúa la validación espacial. El capítulo 5 mide por separado los dos escenarios de predicción. En **zonas conocidas** el orden se invierte: los árboles superan a la logística y el XGBoost geográfico mejora de forma consistente a su control, aunque empata con votar con el estrato de las viviendas vecinas. En **zonas nuevas**, agregar variables de vecindad física da un resultado mixto: mejora el test, pero la validación cruzada no lo confirma.

## Relación con el Entregable 1

El EDA completo, la limpieza y el diseño de la validación se documentaron en el Entregable 1. Para el modelo base, el dashboard **no reentrena ni rehace decisiones**: usa la misma partición, los mismos hiperparámetros y las mismas semillas. Los modelos avanzados del capítulo 4 sí se ajustan en este entregable, con la misma partición y los mismos folds. El script `preparar_datos.py` reproduce las métricas del Entregable 1 al tercer decimal (F1 de test 0.450 para el modelo principal) antes de guardar los resultados que muestra el tablero.

```{note}
**Cómo leer este libro.** El capítulo 1 corresponde a la pestaña Contexto, el 2 a la pestaña EDA y los capítulos 3 a 5 a la pestaña ML models. Las figuras son las mismas, generadas por el mismo módulo (`figuras.py`), así que lo que se ve en el tablero y lo que se interpreta aquí coincide exactamente. El código de cada figura está oculto; se puede desplegar con el botón de cada celda.
```
