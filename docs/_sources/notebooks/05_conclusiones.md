# 5. Conclusiones y próximos pasos

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

## 5.1 Qué se aprendió

**El estrato se puede predecir con el catastro, pero solo en parte.** Una regresión logística con las características físicas de la vivienda y su ubicación alcanza un F1 macro de 0.30 en validación cruzada espacial y de 0.45 en siete bloques de test, con errores casi siempre de un estrato (accuracy ±1 = 0.94; kappa cuadrático = 0.81). Supera con claridad a las líneas base Dummy y, en todas las métricas de test, a la regla geográfica "clase más frecuente en la zona", aunque esta última diferencia de F1 no es estadísticamente significativa.

**La ubicación es la información dominante.** La posición norte-sur es la variable más asociada al estrato y la de mayor peso en el modelo. Barranquilla está segregada: el estrato medio pasa de 1.5 en el sur a cerca de 5 en el norte, aunque la franja más al norte vuelve a mezclar estratos. Entre las variables físicas, el número de baños, el piso de ubicación y el área construida son las más informativas; el régimen de propiedad separa bien los predios informales.

**La estructura de los datos manda sobre la cantidad de filas.** El estrato casi no varía dentro de un edificio (ICC = 0.991) y tiene una autocorrelación espacial muy fuerte (Moran = 0.91). Las 332 718 viviendas equivalen, en precisión, a unas 1 870 observaciones independientes. Validar con filas mezcladas habría duplicado el F1 (0.60 frente a 0.30), así que el protocolo de bloques con buffer es lo que hace creíbles las cifras de este informe.

**El modelo lineal llegó a su techo, pero los modelos más complejos no lo superan en zonas nuevas.** La curva de aprendizaje es casi plana hasta 54 000 viviendas y los residuos siguen autocorrelacionados (Moran = 0.70): el modelo capta la tendencia de la ciudad, pero no los barrios. Sin embargo, al probar los candidatos de la revisión bibliográfica con el mismo protocolo (capítulo 4), ninguno mejoró a la logística. La SVM con kernel RBF queda cerca (F1 de test 0.43 frente a 0.45) y reconoce mejor los estratos altos; XGBoost y Rotation Forest quedan por debajo en test; y la adaptación del **XGBoost geográfico a clasificación**, la propuesta novedosa del proyecto, no aporta frente a su control porque los modelos locales no tienen datos en las zonas nuevas que evalúa la validación espacial.

## 5.2 Limitaciones

- **Cobertura.** El 16 % de las viviendas (predios informales, casi todos de estratos 1 y 2) no tiene coordenadas y no entra al modelo. Las conclusiones aplican a la ciudad formal.
- **Pocos bloques de test.** Con siete bloques, los intervalos son muy anchos y el resultado de test es favorable (pocos estratos 6 y un modelo final con más datos que cada fold). La cifra más representativa es la de la CV espacial.
- **Buffer parcial.** El alcance del correlograma (2.9 km) es mayor que el buffer usado (1 km): la validación sigue siendo algo optimista.
- **El estrato es una etiqueta administrativa.** Se asigna por manzana con información del entorno (vías, equipamientos, servicios) que el catastro no contiene.
- **Calidad del catastro.** Las características físicas pueden estar desactualizadas; hay años de construcción amontonados en ciertos valores y viviendas con 0 habitaciones o 0 baños que no se pudieron verificar con un diccionario oficial.
- **Riesgo ético.** Un modelo que predice el estrato podría usarse para discriminar (crédito, seguros). Su propósito aquí es analítico. En el dashboard las coordenadas se redondean a unos 110 m y no se publica el número predial.

## 5.3 Próximos pasos

Los resultados del capítulo 4 cambian la agenda: el cuello de botella no es la flexibilidad del modelo, sino la información disponible en zonas sin datos y la forma de evaluar.

1. **Dos escenarios de predicción.** Evaluar por separado la predicción en **zonas nuevas** (validación por bloques, como aquí) y **dentro de zonas conocidas** (viviendas ocultas al azar dentro de los bloques de entrenamiento). El XGBoost geográfico {cite}`grekousis2025` está pensado para el segundo escenario; ahí es donde su adaptación a clasificación debería medirse.
2. **Combinar la logística y la SVM RBF** (*stacking*), porque aciertan estratos distintos en la parte alta de la escala: la SVM reconoce mejor el 5 y la logística el 4.
3. **Regresión logística ordinal** como segunda referencia lineal, porque respeta el orden de los estratos.
4. **Variables de vecindad física** (área, baños o antigüedad promedio de los vecinos, a cualquier distancia), que no usan el estrato y por tanto no son fuga, y que sí están disponibles en zonas nuevas.
5. **Recalibración** de probabilidades y umbrales por clase, dentro de la CV espacial.
6. **Búsquedas de hiperparámetros más amplias** (más ensayos TPE, Bagged TAO Trees {cite}`carreira2020`) cuando se disponga de más cómputo.
