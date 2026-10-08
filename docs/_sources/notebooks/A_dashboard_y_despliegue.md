# A. El dashboard y su despliegue

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

## A.1 Arquitectura

El proyecto separa el cálculo pesado de la visualización, para que el tablero sea rápido y quepa en el plan gratuito de Render (512 MB de memoria):

| Archivo | Qué hace |
|---|---|
| `preparar_datos.py` | Se ejecuta una vez, en local. Lee las salidas del Entregable 1, reajusta los seis modelos con los hiperparámetros ya elegidos, verifica que las métricas de test coinciden con las del Entregable 1 y guarda en `datos/` las predicciones, coeficientes, diagnósticos (curva de aprendizaje, sensibilidad al buffer, correlograma, Moran) y el modelo principal. |
| `entrenar_avanzados.py`, `avanzados.py` | Entrenan y evalúan los modelos del capítulo 4 (XGBoost, XGBoost geográfico, Rotation Forest, SVM RBF + TPE) con el mismo protocolo, y agregan sus resultados a `datos/`. Necesitan además `requirements-entrenamiento.txt` (xgboost y optuna); el dashboard no. |
| `figuras.py` | Única fuente de las figuras (Plotly). La usan el dashboard y este libro. |
| `app.py` | La aplicación Dash: tres pestañas (Contexto, EDA y ML models, esta última con siete sub-pestañas), filtros y callbacks. No entrena nada. |
| `generar_portada.py` | Genera, a partir de los datos, las imágenes de `assets/`: la portada «Barranquilla de noche» y la infografía de la escala de estratos. |
| `vecindad/` | Código del capítulo 5 (variables de vecindad y escenario de zonas conocidas); sus resultados están en `datos/vecindad/` y el dashboard solo los grafica. |
| `src/` | Módulos del Entregable 1 (preprocesamiento, partición, métricas). La app los necesita para cargar el modelo guardado. |
| `assets/` | Hoja de estilos (tema Bootstrap Flatly y estilo propio, con letra grande) e imágenes generadas con los datos. |
| `requirements.txt`, `render.yaml`, `.python-version` | Dependencias con versiones fijadas y configuración de Render. |

## A.2 Interactividad

| Pestaña | Controles |
|---|---|
| Contexto | Estática: portada generada con los datos, el problema y su importancia (escala de estratos y tarifas), ficha técnica, brecha de la literatura, flujo de trabajo y hoja de ruta. |
| EDA | Hallazgos clave; partición (solo entrenamiento o todas); sub-pestañas de variable objetivo, numéricas (variable y escala log), categóricas, correlaciones y faltantes, y componente espacial (variable del mapa). |
| ML models | Sub-pestañas: Protocolo; Modelo base (métrica de la comparación, coeficientes de A o B); Diagnóstico (modelo y matriz en conteos o porcentaje); Modelos de la revisión (métrica, mapa de residuos por modelo); Vecindad y escenarios; Simulador; Conclusiones. |

## A.3 Datos que se publican

El dashboard usa el **conjunto completo** de 332 718 viviendas. Para reducir el riesgo de reidentificación se eliminó el número predial (se reemplazó por un identificador anónimo de edificio) y las coordenadas se redondearon a tres decimales (unos 110 m). Los mapas muestran promedios por celda, no viviendas individuales.

## A.4 Despliegue en Render

1. Subir la carpeta del dashboard a un repositorio de GitHub (con `datos/`, `src/`, `assets/` y los archivos de configuración).
2. En Render, crear un **Web Service** conectado a ese repositorio.
3. Build command: `pip install -r requirements.txt`. Start command: `gunicorn app:server --workers 1 --threads 2 --timeout 120`.
4. Plan: Free. Render asigna una dirección del tipo `https://<nombre>.onrender.com`.

En el plan gratuito el servicio se suspende tras unos minutos sin visitas; la primera carga después de eso tarda alrededor de un minuto.

## A.5 Reproducibilidad

- Semilla 42 en todo el proyecto (`src/config.py`).
- Versiones fijadas en `requirements.txt`. El modelo guardado con `joblib` requiere la misma versión de scikit-learn (1.8.0).
- Para regenerar los datos del dashboard: `python preparar_datos.py ruta/al/entregable1` y luego `python entrenar_avanzados.py ruta/al/entregable1` (unos 40 minutos con 2 núcleos).
- Para reconstruir este libro: `python construir_libro.py` dentro de `libro/`.
