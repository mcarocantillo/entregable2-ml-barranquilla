# Variables de vecindad física y escenario de zonas conocidas

Código del capítulo 5 del libro. Los resultados (JSON) quedan en `datos/vecindad/`.

## Por qué hay una réplica sin scikit-learn

En el equipo donde se corrió, una política de control de aplicaciones de Windows bloquea las librerías compiladas
de scikit-learn. `replica_sin_sklearn.py` reimplementa con numpy y scipy el preprocesamiento y la regresión
logística del Entregable 1 (mismas fórmulas), y usa sin cambios la limpieza, la partición y los folds de su `src/`.
Si scikit-learn sí carga, se usa su `KDTree`; si no, uno equivalente de scipy.

`01_verificar_replica.py` comprueba que la réplica reproduce el modelo B guardado (F1 por fold con diferencias de
0.001 o menos y F1 de test 0.4500). Solo después de esa verificación tienen sentido las comparaciones.

## Requisitos

- Python 3.11 o superior con numpy, pandas, scipy y pyarrow.
- Para `06_zonas_conocidas.py`, además `xgboost` (se usa su API básica, sin scikit-learn).
- La carpeta del Entregable 1 con `src/` y `datos/predios_residenciales_barranquilla.csv`. Por defecto se busca
  `../../entregable1_jbook` (junto al repositorio); se puede indicar con la variable de entorno `ENTREGABLE1`.

## Orden de ejecución

Ejecutar desde esta carpeta:

| Script | Qué hace | Salida en `datos/vecindad/` | Tiempo aprox. |
|---|---|---|---|
| `prueba_variables_vecindad.py` | Verifica el cálculo de la vecindad contra fuerza bruta | (consola) | segundos |
| `01_verificar_replica.py` | Reconstruye datos y partición; verifica la réplica del modelo B | `verificacion_replica.json`, `datos.pkl` | 3 min |
| `02_comparar_modelos.py` | B frente a B + vecindad y variantes; CV espacial de 5 folds con búsqueda y regla 1-SE; test | `comparacion_zonas_nuevas.json`, `pred_test.pkl` | 1.5 a 2 h |
| `03_variante_estricta.py` | Vecindad calculada solo dentro de cada partición; quitar un bloque a la vez | `variante_estricta.json` | 3 min |
| `04_cv_15_folds.py` | CV espacial de 15 folds (solo entrenamiento) | `cv_15_folds.json` | 15 min |
| `05a_predicciones_oof.py` | Guarda las predicciones fuera de muestra de los 15 folds | `oof15.pkl` | 10 min |
| `05b_analisis_por_zonas.py` | Dónde ayuda y dónde perjudica la vecindad (exploratorio) | `zonas.json` | 1 min |
| `06_zonas_conocidas.py` | Escenario 2: edificios ocultos dentro de zonas con datos | `zonas_conocidas.json` | 20 min |

Los archivos `.pkl` son intermedios grandes y no se suben al repositorio (ver `.gitignore`).

## Reglas contra la fuga de datos

- Las variables de vecindad usan solo atributos físicos (área, baños, antigüedad, tipo de vivienda, régimen y
  densidad) y excluyen el propio edificio. Nunca usan el estrato de los vecinos.
- El estrato de los vecinos aparece solo en dos lugares: como diagnóstico en `05b` (nunca entra a un modelo) y en
  el escenario de zonas conocidas (`06`), donde esa información sí estaría disponible en la práctica.
- En el escenario 2 se ocultan edificios completos, nunca filas sueltas (ICC = 0.991).
