# A. Historia de las versiones

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

El proyecto se revisó dos veces después de la primera entrega. Cada versión conserva la anterior publicada (la v1 en la
raíz del sitio, la v2 en `/v2/` y esta en `/v3/`), y el código de cada una está en su rama o etiqueta del repositorio.

| Versión | Qué cambió | Por qué | Efecto principal |
|---|---|---|---|
| **v1** (entregada) | Modelo base y candidatos de la revisión con validación por bloques de 2 km y buffer de 1 km | — | Ningún candidato superó a la logística en test |
| **v2** | Las viviendas sin coordenadas propias (16 %) se ubican con el centroide de su manzana; pesos de clase; vecindad física; escenario de zonas conocidas (ocultando edificios y manzanas) | Las viviendas excluidas eran casi todas de estratos 1 y 2: sesgaban el modelo | Cobertura del 83.8 % al 96.9 %; los árboles ganan en zonas conocidas; la vecindad dio un «resultado mixto» en zonas nuevas |
| **v3** | Validación **kNNDM** en lugar del buffer fijo; vecindad con **grafo KNN**; XGBoost geográfico con vecindad (ronda 2) | Sugerencia del profesor (libro de Modelos Espacio-Temporales, cap. 2): usar vecinos más cercanos | El buffer evaluaba un escenario más difícil que el declarado; con kNNDM la vecindad ayuda y **el modelo nuevo supera al base** |
| **v3 (reorganizada)** | El libro y el dashboard se ordenan como investigación: revisión → hipótesis → método → resultados | El orden por versiones era confuso | Mismas cifras, mejor contadas |

## A.1 Réplica sin scikit-learn (versión 2)

En el equipo donde se corrieron los primeros análisis de vecindad, una política de control de aplicaciones de Windows
bloqueaba las librerías compiladas de scikit-learn. El preprocesamiento y la regresión logística se reimplementaron con
numpy y scipy, con las mismas fórmulas del Entregable 1, y se verificó que la réplica reproduce el modelo base:

| | Réplica | Entregable 1 |
|---|---|---|
| F1 macro por fold (CV espacial) | 0.169, 0.375, 0.357, 0.409, 0.300 | 0.168, 0.375, 0.357, 0.409, 0.300 |
| F1 macro en test | 0.4919 | 0.4921 |
| Accuracy / MAE ordinal / kappa en test | 0.611 / 0.439 / 0.835 | 0.611 / 0.439 / 0.835 |

Los análisis de la versión 3 (`07_vecindad_knn.py` y `08_arboles_knndm.py`) usan ya scikit-learn y el código del
Entregable 1 versión 3, y reproducen el F1 de test del modelo base (0.4932).

## A.2 La validación con buffer (versiones 1 y 2)

Con el modelo base, el F1 de la validación cruzada dependía del tamaño del buffer: 0.41 sin buffer, 0.37 con 0.5 km,
0.32 con 1 km y 0.15 con 2.6 km (el alcance del correlograma). A partir de cierto punto la caída mezclaba dos efectos:
menos fuga por cercanía y menos datos de entrenamiento. La versión 3 mide directamente la distancia que importa (la del
vecino conocido más cercano) y diseña los pliegues para reproducirla (capítulo 5).

## A.3 Qué cifras cambiaron entre versiones

| Cifra | v1 | v2 | v3 |
|---|---|---|---|
| Viviendas con coordenadas | 83.8 % | 96.9 % | 96.9 % |
| F1 del modelo base, validación cruzada | 0.325 | 0.322 | 0.460 (kNNDM) |
| F1 del modelo base, test | 0.454 | 0.492 | 0.493 |
| Mejor modelo en zonas nuevas, test | Logística (0.454) | Logística (0.492) | XGBoost geográfico + vecindad (0.585) |

El test es el mismo conjunto de bloques desde la v2 (en la v1 la partición era distinta porque el universo de
viviendas ubicadas era menor). El cambio de 0.32 a 0.46 en la validación cruzada no es una mejora del modelo: es que la
validación con kNNDM ya no exige predecir a 1.4–1.5 km del dato más cercano.
