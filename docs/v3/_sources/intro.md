# Estrato socioeconómico y catastro en Barranquilla — Entregable 2

**Machine Learning — Maestría, Universidad del Norte · Profesor Lihki Rubio**

**Autores:** María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329) · Octubre de 2026 ·
Versión 3

```{admonition} En una página
:class: tip
**Pregunta.** ¿Se puede predecir el estrato socioeconómico (1 a 6) de una vivienda de Barranquilla con sus datos
catastrales abiertos (área, baños, habitaciones, pisos, antigüedad, régimen de propiedad) y su ubicación?

**Revisión bibliográfica.** Ningún estudio encontrado clasifica el estrato de una vivienda con datos catastrales, y los
nueve modelos con evidencia de reducir el error que lista la revisión **nunca se habían aplicado a este problema**. El
de mejor respaldo para problemas espaciales, el **XGBoost geográfico** {cite}`grekousis2025`, solo existía para
regresión.

**Objetivo.** Usar **modelos no aplicados antes a esta aplicación**, encabezados por el XGBoost geográfico **adaptado
por nosotros a clasificación**, y comprobar si alguno supera al **modelo base**: una regresión logística multinomial
con las variables de la vivienda y su ubicación.

**Respuesta.** **Sí, con condiciones** (capítulos 8 y 9). Con la información del entorno de la vivienda (la vecindad
física), el XGBoost geográfico supera al modelo base en zonas nuevas (F1 macro **0.585 frente a 0.493** en test) y en
zonas con datos cerca lo supera por +0.19 a +0.23. Otros dos modelos originales de la revisión, el **Rotation Forest** y
el **stacking**, también superan al base con significancia estadística; el Rotation Forest es el único con un intervalo
de test que excluye el 0 (+0.099, IC 95 % [+0.024, +0.170]). Pero en zonas nuevas **ninguno le gana a los modelos
convencionales con la misma información**: la mejora la explica sobre todo describir el entorno.
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
| | **8. ¿Supera el modelo nuevo al modelo base?** | **La respuesta para el modelo principal (XGBoost geográfico)** |
| | **9. Todos los modelos originales de la revisión** | **¿Algún otro candidato de la revisión supera al modelo base? ¿Es significativo?** |
| **Cierre** | 10. Conclusiones | Respuesta, limitaciones y próximos pasos |
| **Apéndices** | A. Historia de las versiones · B. Dashboard · C. Referencias | Qué cambió de la v1 a la v3, cómo está hecho el tablero y la bibliografía |

```{note}
**El dashboard.** El libro acompaña un tablero en Dash Plotly con la misma historia, en ocho pestañas: **Problema y
revisión**, **Datos y calidad**, **Análisis exploratorio**, **Modelo base**, **Modelos propuestos y protocolo**,
**Resultados**, **Optimización (Entregable 3)** y **Conclusiones**. Está en
**https://entregable2-ml-barranquilla.onrender.com** (el servicio gratuito se suspende sin visitas: la primera carga
puede tardar un minuto). Las figuras de los dos son las mismas, generadas por los módulos `figuras.py`,
`figuras_proyecto.py` y `figuras_mejoras.py`. El código de cada figura está oculto; se despliega con el botón de
cada celda.
```

**Relación con el Entregable 1.** La limpieza, el EDA completo, la partición y el modelo base se documentaron en el
Entregable 1 (versión 3: https://mcarocantillo.github.io/entregable1-ml-barranquilla/v3/). Este entregable usa la
misma partición y las mismas semillas, y reproduce sus métricas antes de comparar.
