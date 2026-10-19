# 10. Conclusiones

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

## 10.1 Respuesta a la pregunta y a la hipótesis

**Pregunta.** ¿Se puede predecir el estrato socioeconómico de una vivienda de Barranquilla con su información catastral
abierta y su ubicación? **En parte.** El modelo base (regresión logística) acierta el estrato exacto en algo más de la
mitad de los casos de una zona nueva (F1 macro 0.49 en test) y casi siempre a un estrato de distancia (accuracy ±1 de
0.96). La ubicación domina, como anticipa la metodología del DANE: el estrato es una propiedad de la zona.

**Brecha de la revisión bibliográfica.** Ningún estudio encontrado clasifica el estrato de una vivienda con datos
catastrales (brecha de aplicación), y los nueve modelos con evidencia de reducir el error que lista la revisión nunca se
habían aplicado a este problema (brecha metodológica). El de mejor respaldo para problemas espaciales, el XGBoost
geográfico, solo existía para regresión.

**Hipótesis: un modelo no usado antes en esta aplicación supera al modelo base. Se confirma, con condiciones.**

| | Resultado |
|---|---|
| Ronda 1 (variables del modelo base, validación con buffer) | Ningún candidato original lo supera en el test: XGBoost geográfico, Rotation Forest y SVM con TPE (cap. 6) |
| Ronda 2, zonas nuevas, con vecindad física: XGBoost geográfico (cap. 8) | Lo supera en validación (5 de 5 pliegues) y en test: 0.585 frente a 0.493 (+0.092; IC por percentiles [+0.034, +0.199], IC BCa [−0.024, +0.172]) |
| Ronda 2, zonas nuevas, todos los candidatos con vecindad (cap. 9) | **XGBoost geográfico, Rotation Forest y stacking lo superan con significancia** (30 bloques, Holm, p < 0.01); el **Rotation Forest** es el único con IC de test que excluye el 0 (+0.099 [+0.024, +0.170]) |
| Ronda 2, zonas conocidas: XGBoost geográfico (cap. 8) | **Lo supera por +0.19 a +0.23 de F1**, y también a una logística con la misma información |

1. **Necesitan información del entorno.** En zonas sin datos, la ventaja de los modelos originales aparece cuando
   reciben la vecindad física (cómo son las viviendas de alrededor, sin usar su estrato).
2. **Con esa misma información, ningún modelo original le gana a los convencionales.** La logística, XGBoost y Random
   Forest con la vecindad física empatan con ellos (ninguna diferencia significativa): en zonas nuevas, la mejora la
   explica sobre todo la información, no el algoritmo.
3. **La ventaja propia del XGBoost geográfico aparece donde hay datos cerca.** En zonas conocidas, sus modelos
   locales aprenden el mapa del barrio y superan incluso a la logística con la misma información (+0.02 a +0.08).

## 10.2 Qué se aprendió en el camino

- **Cómo se valida cambia las conclusiones.** Un buffer fijo de 1 km evaluaba a 1.4–1.5 km del dato más cercano, más
  lejos que el escenario declarado (0.45 km). Con la validación kNNDM, que reproduce esa distancia, la validación
  cruzada anticipa el test (0.46 frente a 0.49) y la vecindad física pasa de «resultado mixto» a mejora significativa.
- **El escenario importa.** El mejor modelo para zonas nuevas y para zonas conocidas no es el mismo, y hay que
  reportarlos por separado.
- **Los demás candidatos de la revisión.** El Rotation Forest pasó de tener solo el menor error ordinal (ronda 1) a ser
  el modelo original más sólido con la vecindad física (cap. 9). La SVM con TPE pasó el criterio de validación pero no
  se sostuvo en test; el RF + SVM tampoco se confirma en test; el stacking empata con el mejor en validación.
- **Afinar no siempre mejora.** Buscar los hiperparámetros del XGBoost geográfico con todos los datos subió su
  validación (0.581) pero no su test (0.566 frente a 0.585).
- **La regresión logística ordinal y los umbrales por estrato no mejoraron** al modelo base.

## 10.3 Limitaciones

- **Pocos bloques de test (7).** Los intervalos son anchos y con 7 bloques ninguna diferencia puede ser significativa
  después de corregir por comparaciones múltiples; las pruebas con más réplicas usan los 30 bloques espaciales de la
  validación (cap. 9).
- **Candidatos no probados.** Bagged TAO Trees (sin implementación abierta), y los métodos de optimización y de
  selección de variables de la revisión (GWO, EFSHB, NSGA-II).
- **Cobertura.** Quedan sin ubicar 10 300 viviendas (3.1 %), casi todas de estrato 1; las ubicadas por manzana
  comparten coordenada.
- **El estrato es una etiqueta administrativa** que usa insumos no publicados (zonas homogéneas, puntaje de las
  edificaciones) y ajustes manuales; el modelo tiene un techo que no depende de su flexibilidad.
- **Riesgo ético.** Un modelo que predice el estrato podría usarse para discriminar (crédito, seguros). Su propósito
  aquí es analítico; el dashboard redondea las coordenadas a unos 110 m y no publica el número predial.

## 10.4 Próximos pasos

1. ~~Búsqueda de hiperparámetros del XGBoost geográfico con la validación kNNDM~~ y comparación estadística formal con
   todos los modelos de la guía: **hechas** (capítulo 9 y Entregable 3).
2. **Medir la variedad física del entorno** (dispersión del área o de los baños a 300 m), para que el modelo sepa
   cuándo confiar en la vecindad.
3. **Un modelo en dos etapas, como el del DANE:** estimar el estrato de la zona y luego decidir si cada vivienda es
   atípica frente a ella; e incorporar las zonas homogéneas o el puntaje de calificación si la Alcaldía los entrega.
4. ~~Combinar modelos (stacking)~~ **hecho** (capítulo 9); queda probar Bagged TAO Trees {cite}`carreira2020` cuando
   haya una implementación disponible.
5. **Más zonas nuevas de test** (otra partición u otra ciudad), para poder confirmar diferencias pequeñas.
