# Entregable 2 — Dashboard y Jupyter Book

**Estrato socioeconómico de las viviendas de Barranquilla a partir del catastro**

Machine Learning — Maestría, Universidad del Norte · Profesor Lihki Rubio

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

## Qué hay en esta carpeta

| Ruta | Contenido |
|---|---|
| `app.py` | Dashboard en Dash Plotly con 3 pestañas: contexto, EDA y modelos base (con una sección sobre los modelos de la revisión bibliográfica) |
| `figuras.py` | Todas las figuras (las usan el dashboard y el libro) |
| `preparar_datos.py` | Genera `datos/` a partir del Entregable 1 (ya está ejecutado) |
| `entrenar_avanzados.py`, `avanzados.py` | Entrenan y evalúan los modelos de la revisión bibliográfica (XGBoost geográfico adaptado a clasificación, XGBoost, Rotation Forest, SVM RBF + TPE); ya están ejecutados. Para volver a correrlos se necesita además `requirements-entrenamiento.txt` |
| `datos/` | Datos y resultados que usa el dashboard (332 718 viviendas, sin número predial); en `datos/vecindad/`, los resultados del capítulo 5 |
| `vecindad/` | Capítulo 5: variables de vecindad física y escenario de zonas conocidas (ver su `README.md`) |
| `src/` | Módulos del Entregable 1 (necesarios para cargar el modelo) |
| `assets/` | Estilos del dashboard |
| `requirements.txt`, `render.yaml`, `.python-version` | Configuración para Render |
| `libro/` | Fuentes del Jupyter Book (cuadernos, referencias, configuración) |
| `docs/` | El Jupyter Book ya construido (HTML), listo para GitHub Pages |

## Probar el dashboard en tu computador (opcional)

```
pip install -r requirements.txt
python app.py
```

Luego abre http://localhost:8050 en el navegador.

## Publicar en GitHub y Render

**Paso 1. Crear el repositorio.** En GitHub, crea un repositorio nuevo y público (por ejemplo, `entregable2-estrato`). Sube **todo el contenido de esta carpeta**, incluidas `datos/`, `src/`, `assets/`, `libro/` y `docs/`, y los archivos que empiezan con punto (`.python-version`, `.gitignore`, `docs/.nojekyll`).

**Paso 2. Crear el servicio en Render.** En https://render.com entra con tu cuenta de GitHub. Elige **New → Web Service** y conecta el repositorio. Completa así:

- Name: `estrato-barranquilla`
- Language: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:server --workers 1 --threads 2 --timeout 120`
- Instance type: Free

Pulsa **Create Web Service** y espera a que el registro diga "Your service is live". La dirección será `https://entregable2-ml-barranquilla.onrender.com`. Si Render te asigna otra, cámbiala en `libro/intro.md` y en `docs/intro.html`.

**Paso 3. Publicar el libro.** En el repositorio de GitHub ve a **Settings → Pages**. En "Branch" elige `main` y la carpeta `/docs`, y guarda. En uno o dos minutos el libro queda en `https://mcarocantillo.github.io/entregable2-ml-barranquilla/`.

## Reconstruir el libro (solo si cambias algo)

Dentro de `libro/`:

```
python construir_libro.py
```

Después copia `libro/_build/html/` sobre `docs/` (conservando `docs/.nojekyll`).

## Reproducibilidad

- Semilla 42 en todo el proyecto (`src/config.py`).
- Versiones fijadas en `requirements.txt`; el modelo guardado necesita scikit-learn 1.8.0.
- `preparar_datos.py` reproduce las métricas del Entregable 1 (F1 de test 0.450 del modelo principal) antes de guardar los resultados.
