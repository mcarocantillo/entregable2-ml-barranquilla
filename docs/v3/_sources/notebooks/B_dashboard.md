# B. El dashboard y su despliegue

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

## B.1 Arquitectura

El proyecto separa el cálculo pesado de la visualización, para que el tablero sea rápido y quepa en el plan gratuito de Render (512 MB de memoria):

| Archivo | Qué hace |
|---|---|
| `preparar_entregable1.py` | Resume el capítulo 1 del Entregable 1 (embudo, cobertura de coordenadas, faltantes, ICC y efecto de diseño) en `datos/entregable1.json`, solo con agregados. La pestaña «Base de datos y calidad» lo necesita. |
| `preparar_datos.py` | Se ejecuta una vez, en local. Lee las salidas del Entregable 1, reajusta los seis modelos con los hiperparámetros ya elegidos, verifica que las métricas de test coinciden con las del Entregable 1 y guarda en `datos/` las predicciones, coeficientes, diagnósticos (curva de aprendizaje, sensibilidad al buffer, correlograma, Moran) y el modelo principal. |
| `entrenar_avanzados.py`, `avanzados.py` | Entrenan y evalúan los candidatos de la ronda 1 (capítulos 4 y 6: XGBoost, XGBoost geográfico, Rotation Forest, SVM RBF + TPE) con el mismo protocolo, y agregan sus resultados a `datos/`. Necesitan además `requirements-entrenamiento.txt` (xgboost y optuna); el dashboard no. |
| `figuras.py` | Figuras del Entregable 2 (Plotly). Las usan el dashboard y este libro. |
| `figuras_proyecto.py` | Figuras de los capítulos del Entregable 1, la vecindad y los escenarios, para el dashboard final. |
| `proyecto/` | Precalculan `datos/proyecto/`: tablas de los cuadernos del Entregable 1, PCA, outliers, K-Means, t-SNE, LISA, mapa de la partición, búsqueda de C y recuperación por manzana. |
| `app.py`, `diapositivas.py`, `componentes.py`, `glosario.py` | El dashboard final: portada, siete pestañas de diapositivas (vista continua o por diapositiva), glosario de métricas y modo presentación. No entrena nada. |
| `generar_portada.py` | Genera la infografía de la escala de estratos. |
| `vecindad/` | Ronda 2 (capítulos 7 y 8): variables de vecindad, escenario de zonas conocidas, `07_vecindad_knn.py` (grafo KNN y validación kNNDM) y `08_arboles_knndm.py` (XGBoost y XGBoost geográfico con vecindad y kNNDM). Resultados en `datos/vecindad/`, en la pestaña «Resultados». |
| `src/` | Módulos del Entregable 1 (preprocesamiento, partición, métricas). La app los necesita para cargar el modelo guardado. |
| `assets/` | Hoja de estilos (tema Bootstrap Flatly y estilo propio), navegación con el teclado, la infografía de estratos y la imagen satelital de la portada (Copernicus Sentinel-2, 2021, procesada por la ESA). |
| `requirements.txt`, `render.yaml`, `.python-version` | Dependencias con versiones fijadas y configuración de Render. |

## B.2 Organización e interactividad

El dashboard final reúne los dos entregables y los análisis posteriores con la misma historia de este libro. Tiene una
portada y siete pestañas (Problema y revisión, Datos y calidad, Análisis exploratorio, Modelo base, Modelo propuesto y
protocolo, Resultados y Conclusiones); en la **vista continua** cada pestaña muestra todas sus diapositivas una debajo
de otra, con un menú lateral que salta a cada una y flechas del teclado, y en la vista **por diapositiva** se pasan de
a una. Cada diapositiva tiene una gráfica o tabla interactiva, de dos a cuatro frases de lectura,
una conclusión en una frase y **fichas de lectura** de las métricas que usa: qué mide cada una, su rango y una escala
de colores con los umbrales habituales, con el valor del proyecto marcado. El botón «Glosario» reúne todas las fichas
con un buscador, y el botón «Presentar» oculta la portada y el menú para proyectar.

| Pestaña | Contenido y controles |
|---|---|
| 1. Contexto | La pregunta y la escala de estratos; cómo asigna el DANE el estrato; los datos y el número predial; la revisión bibliográfica; la ruta de trabajo. |
| 2. Datos y calidad | Embudo de limpieza; ubicación por manzana (umbral ajustable y mapa); cobertura y modelo v1 frente a v2; faltantes e imposibles; outliers; dependencia y tamaño efectivo (ICC ajustable); partición espacial (train/test/buffer, folds o mezcla de estratos). |
| 3. Análisis exploratorio | Objetivo (distribución, sur–norte, mapa); numéricas (variable y escala); transformación; correlaciones (Spearman, Pearson o pares); densidad de pares (dos variables); numéricas y categóricas frente al estrato; ranking de asociación; VIF; PCA, biplot (componentes y color); outliers multivariados; clustering (k ajustable) y t-SNE; mapa por variable; autocorrelación, LISA y Gi*; escala (MAUP); auditoría de fuga. |
| 4. Modelo base | Protocolo de validación; búsqueda de hiperparámetros; comparación con líneas base (métrica, gráfica o tabla); significancia; errores por estrato, ROC y calibración (modelo); coeficientes y odds; importancia por permutación; curva de aprendizaje; residuos; simulador (variables y clic en el mapa). |
| 5. Más allá del modelo base | Modelos de la revisión (métrica, tabla); criterio de selección (CV frente a test, por fold); errores y residuos por modelo; XGBoost geográfico; vecindad física (zonas nuevas, 15 folds, dónde ayuda); zonas conocidas (ocultando edificios o manzanas) y los dos escenarios. |
| 6. Conclusiones | Hallazgos, respuesta a la pregunta, limitaciones, próximos pasos y enlaces. |

## B.3 Datos que se publican

El dashboard usa el **conjunto completo** de 332 718 viviendas. Para reducir el riesgo de reidentificación se eliminó el número predial (se reemplazó por un identificador anónimo de edificio) y las coordenadas se redondearon a tres decimales (unos 110 m). Los mapas muestran promedios por celda, no viviendas individuales.

## B.4 Despliegue en Render

1. Subir la carpeta del dashboard a un repositorio de GitHub (con `datos/`, `src/`, `assets/` y los archivos de configuración).
2. En Render, crear un **Web Service** conectado a ese repositorio.
3. Build command: `pip install -r requirements.txt`. Start command: `gunicorn app:server --workers 1 --threads 2 --timeout 120`.
4. Plan: Free. Render asigna una dirección del tipo `https://<nombre>.onrender.com`.

En el plan gratuito el servicio se suspende tras unos minutos sin visitas; la primera carga después de eso tarda alrededor de un minuto.

## B.5 Reproducibilidad

- Semilla 42 en todo el proyecto (`src/config.py`).
- Versiones fijadas en `requirements.txt`. El modelo guardado con `joblib` requiere la misma versión de scikit-learn (1.8.0).
- Para regenerar los datos del dashboard: `python preparar_entregable1.py ruta/al/entregable1`, `python preparar_datos.py ruta/al/entregable1` y luego `python entrenar_avanzados.py ruta/al/entregable1`.
- Para reconstruir este libro: `python construir_libro.py` dentro de `libro/`.
