# 6. Conclusiones y próximos pasos

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

## 6.1 Qué se aprendió

**El estrato se puede predecir con el catastro, pero solo en parte.** Una regresión logística con las características físicas de la vivienda y su ubicación alcanza un F1 macro de 0.30 en validación cruzada espacial y de 0.45 en siete bloques de test, con errores casi siempre de un estrato (accuracy ±1 = 0.94; kappa cuadrático = 0.81). Supera con claridad a las líneas base Dummy y, en todas las métricas de test, a la regla geográfica "clase más frecuente en la zona", aunque esta última diferencia de F1 no es estadísticamente significativa.

**La ubicación es la información dominante.** La posición norte-sur es la variable más asociada al estrato y la de mayor peso en el modelo. Barranquilla está segregada: el estrato medio pasa de 1.5 en el sur a cerca de 5 en el norte, aunque la franja más al norte vuelve a mezclar estratos. Entre las variables físicas, el número de baños, el piso de ubicación y el área construida son las más informativas; el régimen de propiedad separa bien los predios informales.

**La estructura de los datos manda sobre la cantidad de filas.** El estrato casi no varía dentro de un edificio o conjunto (ICC = 0.992) y tiene una autocorrelación espacial muy fuerte (Moran = 0.91). Las 332 718 viviendas equivalen, en precisión, a unas 2 300 observaciones independientes. Validar con filas mezcladas habría duplicado el F1 (0.60 frente a 0.30), así que el protocolo de bloques con buffer es lo que hace creíbles las cifras de este informe.

**El modelo lineal llegó a su techo, pero los modelos más complejos no lo superan en zonas nuevas.** La curva de aprendizaje es casi plana hasta 54 000 viviendas y los residuos siguen autocorrelacionados (Moran = 0.70): el modelo capta la tendencia de la ciudad, pero no los barrios. Sin embargo, al probar los candidatos de la revisión bibliográfica con el mismo protocolo (capítulo 4), ninguno mejoró a la logística. La SVM con kernel RBF queda cerca (F1 de test 0.43 frente a 0.45) y reconoce mejor los estratos altos; XGBoost y Rotation Forest quedan por debajo en test; y la adaptación del **XGBoost geográfico a clasificación**, la propuesta novedosa del proyecto, no aporta frente a su control porque los modelos locales no tienen datos en las zonas nuevas que evalúa la validación espacial.

**El mejor modelo depende del escenario de predicción (capítulo 5).** Se midieron dos escenarios por separado. En
**zonas nuevas**, agregar variables de vecindad física (cómo son las viviendas de alrededor, sin usar su estrato)
mejora el test (F1 de 0.45 a 0.59), pero la validación cruzada no lo confirma: su efecto depende de la zona, ayuda en
barrios homogéneos y perjudica en los muy mezclados. En **zonas conocidas** (edificios ocultos dentro de zonas con
datos), los árboles superan a la logística por +0.20 de F1, y el XGBoost geográfico supera a su control en los cinco
folds (+0.010, IC [+0.007, +0.024]). En su escenario natural, la propuesta novedosa sí aporta, aunque poco, pero
empata con la regla más simple: votar con el estrato de las viviendas conocidas más cercanas (F1 0.79).

**Los resultados reproducen la lógica de la metodología del DANE.** Según la metodología vigente {cite}`dane2015conceptual,dane2015manual`, el estrato se asigna a zonas homogéneas del catastro y solo cambia en las viviendas atípicas (sección 1.1.1). Eso explica cuatro hallazgos del informe: que la ubicación domine, que el estrato casi no varíe dentro de un edificio o conjunto (ICC = 0.992), que los vecinos conocidos sean el mejor predictor y que los errores sean casi siempre de un estrato, el mismo salto que la metodología permite a una vivienda atípica. También explica el techo del modelo en zonas nuevas: el catastro abierto no publica las zonas homogéneas ni el puntaje de calificación de las edificaciones, que son los dos insumos con los que el DANE calcula el estrato. La respuesta a la pregunta del proyecto es, entonces, que **la parte pública del catastro revela sobre todo la zona; lo que distingue a la vivienda dentro de su zona no está publicado**.

## 6.2 Limitaciones

- **Cobertura.** El 16 % de las viviendas (predios informales, casi todos de estratos 1 y 2) no tiene coordenadas y no entra al modelo. Las conclusiones aplican a la ciudad formal.
- **Pocos bloques de test.** Con siete bloques, los intervalos son muy anchos y el resultado de test es favorable (pocos estratos 6 y un modelo final con más datos que cada fold). La cifra más representativa es la de la CV espacial.
- **Buffer parcial.** El alcance del correlograma (2.9 km) es mayor que el buffer usado (1 km): la validación sigue siendo algo optimista.
- **El estrato es una etiqueta administrativa.** Se asigna a zonas homogéneas del catastro (vías, servicios, uso y valor del suelo) con el puntaje de calificación de las edificaciones, y lo revisan la alcaldía y el Comité Permanente de Estratificación, que pueden ajustarlo a mano. Ninguno de estos insumos está en los datos abiertos. Además, la estratificación vigente de Barranquilla puede ser anterior a la metodología de 2015 o estar desactualizada, que el DANE señala como la principal fuente de error. Por todo esto, el modelo tiene un techo que no depende de su flexibilidad.
- **Calidad del catastro.** Las características físicas pueden estar desactualizadas; hay años de construcción amontonados en ciertos valores y viviendas con 0 habitaciones o 0 baños que no se pudieron verificar con un diccionario oficial.
- **Riesgo ético.** Un modelo que predice el estrato podría usarse para discriminar (crédito, seguros). Su propósito aquí es analítico. En el dashboard las coordenadas se redondean a unos 110 m y no se publica el número predial.

## 6.3 Próximos pasos

Los resultados de los capítulos 4 y 5 cambian la agenda: el cuello de botella no es la flexibilidad del modelo, sino la información disponible en zonas sin datos y la forma de evaluar. Los dos escenarios de predicción ya se midieron por separado (capítulo 5); quedan estos pasos:

1. **Medir la variedad física del entorno** (por ejemplo, la dispersión del área o de los baños a 300 m). Las variables de vecindad ayudan en barrios homogéneos y perjudican en los muy mezclados; estas medidas, que tampoco usan el estrato, le permitirían al modelo saber cuándo confiar en la vecindad.
2. **Combinar la logística y la SVM RBF** (*stacking*), porque aciertan estratos distintos en la parte alta de la escala: la SVM reconoce mejor el 5 y la logística el 4.
3. **Regresión logística ordinal** como segunda referencia lineal, porque respeta el orden de los estratos.
4. **Más bloques de evaluación.** Con 7 bloques de test y 30 de entrenamiento, varias diferencias del capítulo 5 no se pueden distinguir del ruido. Bloques más pequeños o una validación con más folds darían comparaciones por pares con más poder.
5. **Un modelo en dos etapas, como el del DANE.** Primero estimar el estrato de la zona y luego decidir si cada vivienda es atípica frente a su zona. Y, si la Alcaldía o el catastro entregan las zonas homogéneas o el puntaje de calificación de las edificaciones, incorporarlos: son los insumos que el DANE usa de forma directa.
6. **Recalibración** de probabilidades y umbrales por clase, dentro de la CV espacial.
7. **Búsquedas de hiperparámetros más amplias** (más ensayos TPE, Bagged TAO Trees {cite}`carreira2020`) cuando se disponga de más cómputo.
