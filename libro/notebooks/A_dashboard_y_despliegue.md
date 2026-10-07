# A. El dashboard y su despliegue

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

## A.1 Arquitectura

El proyecto separa el cálculo pesado de la visualización, para que el tablero sea rápido y quepa en el plan gratuito de Render (512 MB de memoria):

| Archivo | Qué hace |
|---|---|
| `preparar_datos.py` | Se ejecuta una vez, en local. Lee las salidas del Entregable 1, reajusta los seis modelos con los hiperparámetros ya elegidos, verifica que las métricas de test coinciden con las del Entregable 1 y guarda en `datos/` las predicciones, coeficientes, diagnósticos (curva de aprendizaje, sensibilidad al buffer, correlograma, Moran) y el modelo principal. |
| `entrenar_avanzados.py`, `avanzados.py` | Entrenan y evalúan los modelos del capítulo 4 (XGBoost, XGBoost geográfico, Rotation Forest, SVM RBF + TPE) con el mismo protocolo, y agregan sus resultados a `datos/`. Necesitan además `requirements-entrenamiento.txt` (xgboost y optuna); el dashboard no. |
| `figuras.py` | Única fuente de las figuras (Plotly). La usan el dashboard y este libro. |
| `app.py` | La aplicación Dash: tres pestañas, filtros y callbacks. No entrena nada. |
| `src/` | Módulos del Entregable 1 (preprocesamiento, partición, métricas). La app los necesita para cargar el modelo guardado. |
| `assets/` | Hoja de estilos (tema Bootstrap Flatly y estilo propio, con letra grande). |
| `requirements.txt`, `render.yaml`, `.python-version` | Dependencias con versiones fijadas y configuración de Render. |

## A.2 Interactividad

| Pestaña | Controles |
|---|---|
| 1. Contexto | Estática: ficha del problema, indicadores, flujo de trabajo, embudo de limpieza y distribución del estrato. |
| 2. EDA | Partición (solo entrenamiento o todas); variable numérica y escala log; variable categórica; variable del mapa. |
| 3. Modelos | Métrica de la comparación (modelo base y modelos avanzados); mapa de residuos por modelo; modelo para la matriz de confusión, ROC y calibración; matriz en conteos o en porcentaje; coeficientes de A o B; simulador. |

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
