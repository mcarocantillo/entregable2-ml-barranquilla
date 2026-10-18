# 9. Conclusiones

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

## 9.1 Respuesta a la pregunta y a la hipótesis

**Pregunta.** ¿Se puede predecir el estrato socioeconómico de una vivienda de Barranquilla con su información catastral
abierta y su ubicación? **En parte.** El modelo base (regresión logística) acierta el estrato exacto en algo más de la
mitad de los casos de una zona nueva (F1 macro 0.49 en test) y casi siempre a un estrato de distancia (accuracy ±1 de
0.96). La ubicación domina, como anticipa la metodología del DANE: el estrato es una propiedad de la zona.

**Brecha de la revisión bibliográfica.** Ningún estudio encontrado clasifica el estrato de una vivienda con datos
catastrales, y el modelo con mejor respaldo para problemas espaciales, el XGBoost geográfico, solo existía para
regresión.

**Hipótesis: el modelo nuevo supera al modelo base. Se confirma, con dos condiciones** (capítulo 8):

| | Resultado |
|---|---|
| Ronda 1 (variables del modelo base, validación con buffer) | No lo supera en zonas nuevas: empata en validación y pierde en test (cap. 6) |
| Ronda 2, zonas nuevas, con vecindad física (validación kNNDM) | **Lo supera en validación (5 de 5 pliegues) y en test: 0.585 frente a 0.493 (+0.092, IC 95 % [+0.034, +0.199])** |
| Ronda 2, zonas conocidas | **Lo supera por +0.19 a +0.23 de F1**, y también a una logística con la misma información |

1. **Necesita información del entorno.** En zonas sin datos, la ventaja aparece cuando el modelo recibe la vecindad
   física (cómo son las viviendas de alrededor, sin usar su estrato); con esa misma información la logística empata
   con él, así que en este escenario la mejora viene sobre todo de la información.
2. **Su ventaja propia aparece donde hay datos cerca.** En zonas conocidas, los modelos locales aprenden el mapa del
   barrio y superan incluso a la logística con la misma información (+0.02 a +0.08).

## 9.2 Qué se aprendió en el camino

- **Cómo se valida cambia las conclusiones.** Un buffer fijo de 1 km evaluaba a 1.4–1.5 km del dato más cercano, más
  lejos que el escenario declarado (0.45 km). Con la validación kNNDM, que reproduce esa distancia, la validación
  cruzada anticipa el test (0.46 frente a 0.49) y la vecindad física pasa de «resultado mixto» a mejora significativa.
- **El escenario importa.** El mejor modelo para zonas nuevas y para zonas conocidas no es el mismo, y hay que
  reportarlos por separado.
- **Otros candidatos de la revisión.** La SVM con TPE pasó el criterio de validación pero no se sostuvo en test
  (optimismo de la búsqueda en los mismos pliegues); Rotation Forest tuvo el menor error ordinal; XGBoost colapsa la
  escala en zonas nuevas sin información del entorno.
- **La regresión logística ordinal y los umbrales por estrato no mejoraron** al modelo base.

## 9.3 Limitaciones

- **Pocos bloques de test (7).** Los intervalos son anchos; la validación cruzada kNNDM es la estimación más estable.
- **Hiperparámetros de la ronda 2 fijos.** Los árboles usan los elegidos en la ronda 1 (con el buffer); una búsqueda
  con la validación kNNDM podría mejorarlos (es parte del Entregable 3).
- **Comparaciones múltiples.** Se evaluaron varias variantes de variables; la conclusión se apoya en la regla fijada de
  antemano y en el test, y el Entregable 3 aplica correcciones formales (Friedman, Nemenyi, Holm).
- **Cobertura.** Quedan sin ubicar 10 300 viviendas (3.1 %), casi todas de estrato 1; las ubicadas por manzana
  comparten coordenada.
- **El estrato es una etiqueta administrativa** que usa insumos no publicados (zonas homogéneas, puntaje de las
  edificaciones) y ajustes manuales; el modelo tiene un techo que no depende de su flexibilidad.
- **Riesgo ético.** Un modelo que predice el estrato podría usarse para discriminar (crédito, seguros). Su propósito
  aquí es analítico; el dashboard redondea las coordenadas a unos 110 m y no publica el número predial.

## 9.4 Próximos pasos

1. **Búsqueda de hiperparámetros del XGBoost geográfico con la validación kNNDM** y comparación estadística formal
   con todos los modelos de la guía (Entregable 3).
2. **Medir la variedad física del entorno** (dispersión del área o de los baños a 300 m), para que el modelo sepa
   cuándo confiar en la vecindad.
3. **Un modelo en dos etapas, como el del DANE:** estimar el estrato de la zona y luego decidir si cada vivienda es
   atípica frente a ella; e incorporar las zonas homogéneas o el puntaje de calificación si la Alcaldía los entrega.
4. **Combinar modelos** (*stacking* de la logística y la SVM, que aciertan estratos distintos) y probar Bagged TAO
   Trees {cite}`carreira2020` cuando haya más cómputo.
