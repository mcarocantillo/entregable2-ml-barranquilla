# Estrato socioeconómico y catastro en Barranquilla — Entregable 2

**Machine Learning — Maestría, Universidad del Norte · Profesor Lihki Rubio**

**Autores:** María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329) · Octubre de 2026 ·
Versión 3

```{admonition} En una página
:class: tip
**Pregunta.** ¿Se puede predecir el estrato socioeconómico (1 a 6) de una vivienda de Barranquilla con sus datos
catastrales abiertos (área, baños, habitaciones, pisos, antigüedad, régimen de propiedad) y su ubicación?

**Revisión bibliográfica.** Ningún estudio encontrado clasifica el estrato de una vivienda con datos catastrales, y el
modelo con mejor respaldo para problemas espaciales, el **XGBoost geográfico** {cite}`grekousis2025`, solo existía
para regresión.

**Objetivo.** Usar ese modelo, **adaptado por nosotros a clasificación**, y comprobar si supera al **modelo base**: una
regresión logística multinomial con las variables de la vivienda y su ubicación.

**Respuesta.** **Sí, con dos condiciones** (capítulo 8). Con la información del entorno de la vivienda (la vecindad
física), el XGBoost geográfico supera al modelo base en zonas nuevas en validación y en test (F1 macro **0.585 frente
a 0.493**, IC 95 % de la diferencia [+0.034, +0.199]), y en zonas con datos cerca lo supera por +0.19 a +0.23. En
zonas nuevas, una logística con la misma información empata con él; en zonas conocidas, el modelo nuevo la supera.
```

## Cómo está organizado

El libro sigue el orden de una investigación: la pregunta y la revisión, el método y los resultados.

| Parte | Capítulo | Qué responde |
|---|---|---|
| **I. Problema** | 1. El problema y la revisión bibliográfica | ¿Qué se quiere resolver, qué hay en la literatura y qué modelo se propone? |
| | 2. Datos y análisis exploratorio | ¿Qué rasgos de los datos condicionan los modelos y su evaluación? |
| **II. Método** | 3. El modelo base | ¿Contra qué se mide la propuesta? |
| | 4. El modelo propuesto | ¿En qué consiste el XGBoost geográfico adaptado a clasificación? |
| | 5. El protocolo de evaluación | ¿Cómo se decide, sin hacerse trampa, si un modelo es mejor? |
| **III. Resultados** | 6. Ronda 1 | Con las variables del modelo base, ¿supera algún candidato a la logística? |
| | 7. Ronda 2 | ¿Ayuda describir el entorno de la vivienda? |
| | **8. ¿Supera el modelo nuevo al modelo base?** | **La respuesta a la pregunta del proyecto** |
| **Cierre** | 9. Conclusiones | Respuesta, limitaciones y próximos pasos |
| **Apéndices** | A. Historia de las versiones · B. Dashboard · C. Referencias | Qué cambió de la v1 a la v3, cómo está hecho el tablero y la bibliografía |

```{note}
**El dashboard.** El libro acompaña un tablero en Dash Plotly con la misma historia, en siete pestañas: **Problema y
revisión**, **Datos y calidad**, **Análisis exploratorio**, **Modelo base**, **Modelo propuesto y protocolo**,
**Resultados** y **Conclusiones**. Está en **https://entregable2-ml-barranquilla.onrender.com** (el servicio gratuito
se suspende sin visitas: la primera carga puede tardar un minuto). Las figuras de los dos son las mismas, generadas por
los módulos `figuras.py` y `figuras_proyecto.py`. El código de cada figura está oculto; se despliega con el botón de
cada celda.
```

**Relación con el Entregable 1.** La limpieza, el EDA completo, la partición y el modelo base se documentaron en el
Entregable 1 (versión 3: https://mcarocantillo.github.io/entregable1-ml-barranquilla/v3/). Este entregable usa la
misma partición y las mismas semillas, y reproduce sus métricas antes de comparar.
