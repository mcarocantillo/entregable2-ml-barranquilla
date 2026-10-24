# Proyecto final — Dashboard y Jupyter Book (Entregable 2 y versión final)

**Estrato socioeconómico de las viviendas de Barranquilla a partir del catastro**

Machine Learning — Maestría, Universidad del Norte · Profesor Lihki Rubio

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

## Qué hay en esta carpeta

| Ruta | Contenido |
|---|---|
| `app.py` | Dashboard final en Dash Plotly: portada, 4 pestañas (Contexto, EDA, Modelos y Conclusiones, con los Entregables 1 a 4) y glosario de métricas. Cada pestaña es una secuencia de diapositivas con gráficas interactivas; se navega con el menú, los botones o las flechas del teclado, y tiene un modo presentación |
| `diapositivas.py` | Contenido de las diapositivas (visual, lectura breve, conclusión y fichas de métricas) y sus controles |
| `componentes.py`, `glosario.py` | Componentes visuales y el glosario: qué mide cada métrica y cómo saber si su valor es bueno (escala de colores con los umbrales) |
| `figuras.py`, `figuras_proyecto.py`, `figuras_mejoras.py` | Las figuras de los Entregables 1 a 3: las del Entregable 2 (las usa también el libro), las de los capítulos del Entregable 1, la vecindad, los escenarios y el Entregable 3 |
| `diapositivas_e4.py`, `figuras_e4.py` | Entregable 4 (proyecto final): modelo original EPL, validación repetida 5 × 5, pruebas de superioridad, test, conjuntos externos, robustez, costo y predicción en vivo. Los resultados y el modelo serializado están en `datos/e4/`, copiados de [entregable4-ml-barranquilla](https://github.com/mcarocantillo/entregable4-ml-barranquilla) |
| `proyecto/` | Preparan `datos/proyecto/`: tablas de los cuadernos del Entregable 1 (`extraer_tablas_e1.py`), PCA, outliers, K-Means, t-SNE, LISA, mapa de la partición y búsqueda de C (`preparar_proyecto.py`) y recuperación por manzana (`preparar_manzanas.py`); ya están ejecutados |
| `generar_portada.py` | Genera la infografía de la escala de estratos (`assets/escala_estratos.svg`) |
| `correr_local.py` | Arranca el dashboard en local en el puerto indicado |
| `preparar_entregable1.py` | Genera `datos/entregable1.json` (resumen del capítulo 1 del Entregable 1 para la pestaña Base de datos; ya está ejecutado) |
| `preparar_datos.py` | Genera `datos/` a partir del Entregable 1 (ya está ejecutado) |
| `entrenar_avanzados.py`, `avanzados.py` | Entrenan y evalúan los modelos de la revisión bibliográfica (XGBoost geográfico adaptado a clasificación, XGBoost, Rotation Forest, SVM RBF + TPE); ya están ejecutados. Para volver a correrlos se necesita además `requirements-entrenamiento.txt` |
| `datos/` | Datos y resultados que usa el dashboard (332 718 viviendas, sin número predial; el 96.9 % con coordenadas, incluidas las ubicadas por el centroide de su manzana); en `datos/vecindad/`, los resultados del capítulo 5 del libro |
| `vecindad/` | Capítulo 5: variables de vecindad física y escenario de zonas conocidas; `07_vecindad_knn.py`: versión 3 (capítulo 6, grafo KNN y validación kNNDM). Ver su `README.md` |
| `src/`, `e1/`, `estrato3/` | Módulos del Entregable 1 y, para cargar el modelo original del E4, `src/original_model.py`, `src/explicacion.py` y los paquetes `e1` y `estrato3` del E4 (mismas rutas de importación que el archivo serializado) |
| `assets/` | Estilos, navegación con el teclado e imágenes. La portada (`portada_satelite.jpg`) es una imagen de la ESA: *contains modified Copernicus Sentinel data (2021), processed by ESA* |
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

**Paso 3. Publicar el libro.** En el repositorio de GitHub ve a **Settings → Pages**. En "Branch" elige `main` y la carpeta `/docs`, y guarda. En uno o dos minutos el libro queda en `https://mcarocantillo.github.io/entregable2-ml-barranquilla/`. La versión 2 del libro va en `docs/v2/` (se copia ahí `libro/_build/html/`) y queda en `https://mcarocantillo.github.io/entregable2-ml-barranquilla/v2/`; la raíz de `docs/` conserva el libro entregado (versión 1). La versión 3 (validación por vecinos más cercanos, capítulo 6) va en `docs/v3/` y queda en `https://mcarocantillo.github.io/entregable2-ml-barranquilla/v3/`.

## Reconstruir el libro (solo si cambias algo)

Dentro de `libro/`:

```
python construir_libro.py
```

Después copia `libro/_build/html/` sobre `docs/` (conservando `docs/.nojekyll`).

## Reproducibilidad

- Semilla 42 en todo el proyecto (`src/config.py`).
- Versiones fijadas en `requirements.txt`; el modelo guardado necesita scikit-learn 1.8.0.
- `preparar_datos.py` reproduce las métricas del Entregable 1 (F1 de test 0.492 del modelo principal, igual a cuatro decimales) antes de guardar los resultados.
- Orden para regenerar todo: `preparar_entregable1.py`, `preparar_datos.py`, `entrenar_avanzados.py`, los scripts de `vecindad/` (ver su README), los de `proyecto/` (con la ruta al Entregable 1; `extraer_tablas_e1.py` necesita sus cuadernos ejecutados), `generar_portada.py` y `libro/construir_libro.py`.
