"""Glosario de métricas, índices y parámetros del proyecto: qué mide cada uno y cómo leer su valor.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Cada entrada tiene:
- ``nombre`` y ``mide``: qué es, en una o dos frases;
- ``rango``: valores posibles;
- ``tramos``: lista de (desde, hasta, lectura, color) para dibujar una escala de colores. Los cortes son las
  convenciones citadas en ``fuente`` o, si no hay una convención, los valores de referencia de este proyecto
  (por ejemplo, el azar o una línea base);
- ``mejor``: "alto", "bajo" o "" (no es una métrica de desempeño);
- ``nota``: matices de lectura (por qué el umbral no es absoluto, qué comparar);
- ``valor``: función opcional que devuelve el valor de este proyecto, para marcarlo en la escala.

Los cortes de asimetría, curtosis, η², V de Cramér, Pearson frente a Spearman, VIF y AUC univariado son los mismos
que usa el Entregable 1 (``src/estadistica.py``).
"""

V, A, R, G, GR = "#1baf7a", "#eda100", "#e34948", "#2a78d6", "#c3c2b7"  # verde, ámbar, rojo, azul, gris

GLOSARIO = {
    # ------------------------------------------------------------------ desempeño de clasificación
    "f1_macro": dict(
        nombre="F1 macro", grupo="Desempeño",
        mide="Promedio, sin ponderar, del F1 de cada estrato. El F1 de una clase es la media armónica de precisión y "
             "recall. Todas las clases pesan igual, así que un estrato pequeño (el 6) cuenta tanto como el 1.",
        rango="0 (no acierta ninguna clase) a 1 (perfecto)", mejor="alto",
        tramos=[(0, 0.17, "azar o peor", R), (0.17, 0.35, "algo mejor que el azar", A), (0.35, 0.6, "moderado", G),
                (0.6, 0.8, "bueno", V), (0.8, 1, "muy bueno", V)],
        nota="No hay un umbral universal: se compara con las líneas base. Aquí una Dummy que siempre predice la clase "
             "mayoritaria logra 0.09 y una que adivina con las proporciones reales, 0.17.",
        fuente="Referencias del proyecto (Dummy) y escala orientativa"),
    "accuracy": dict(
        nombre="Accuracy (exactitud)", grupo="Desempeño",
        mide="Proporción de viviendas cuyo estrato se predice exactamente.",
        rango="0 a 1", mejor="alto",
        tramos=[(0, 0.35, "no supera a predecir siempre el estrato 1", R), (0.35, 0.6, "moderada", A),
                (0.6, 0.8, "buena", G), (0.8, 1, "muy buena", V)],
        nota="Con clases desbalanceadas engaña: predecir siempre el estrato más frecuente ya da 0.35 en test. Por eso la "
             "métrica principal es el F1 macro."),
    "balanced_accuracy": dict(
        nombre="Accuracy balanceada", grupo="Desempeño",
        mide="Promedio del recall de cada estrato: la accuracy que tendría el modelo si todas las clases fueran del mismo tamaño.",
        rango="0 a 1 (azar con 6 clases ≈ 0.17)", mejor="alto",
        tramos=[(0, 0.17, "azar", R), (0.17, 0.4, "débil", A), (0.4, 0.6, "moderada", G), (0.6, 1, "buena", V)]),
    "precision": dict(
        nombre="Precisión de una clase", grupo="Desempeño",
        mide="De las viviendas que el modelo asigna a un estrato, qué parte es realmente de ese estrato.",
        rango="0 a 1", mejor="alto",
        tramos=[(0, 0.3, "baja", R), (0.3, 0.6, "media", A), (0.6, 0.8, "buena", G), (0.8, 1, "muy buena", V)],
        nota="Se lee junto con el recall: subir uno suele bajar el otro."),
    "recall": dict(
        nombre="Recall (sensibilidad) de una clase", grupo="Desempeño",
        mide="De las viviendas que son de un estrato, qué parte encuentra el modelo.",
        rango="0 a 1", mejor="alto",
        tramos=[(0, 0.3, "bajo", R), (0.3, 0.6, "medio", A), (0.6, 0.8, "bueno", G), (0.8, 1, "muy bueno", V)]),
    "mae_ordinal": dict(
        nombre="MAE ordinal", grupo="Desempeño",
        mide="Error absoluto medio entre el estrato real y el predicho, contado en estratos. Usa el orden: confundir 1 "
             "con 2 cuesta 1 y confundir 1 con 6 cuesta 5.",
        rango="0 (perfecto) a 5", mejor="bajo",
        tramos=[(0, 0.5, "casi siempre en el estrato o a uno", V), (0.5, 1, "errores de un estrato frecuentes", G),
                (1, 1.4, "poco mejor que una línea base", A), (1.4, 5, "como una Dummy o peor", R)],
        nota="Las Dummy de este proyecto tienen un MAE de 1.4 a 1.96."),
    "acc1": dict(
        nombre="Accuracy ±1 estrato", grupo="Desempeño",
        mide="Proporción de predicciones que caen en el estrato real o en uno vecino.",
        rango="0 a 1", mejor="alto",
        tramos=[(0, 0.6, "baja", R), (0.6, 0.85, "media", A), (0.85, 0.95, "alta", G), (0.95, 1, "muy alta", V)]),
    "kappa": dict(
        nombre="Kappa cuadrático (Cohen ponderado)", grupo="Desempeño",
        mide="Acuerdo entre estrato real y predicho corregido por el azar, con penalización cuadrática por la distancia "
             "entre estratos (ideal para clases ordinales).",
        rango="−1 a 1 (0 = acuerdo de azar)", mejor="alto",
        tramos=[(-1, 0, "peor que el azar", R), (0, 0.2, "leve", R), (0.2, 0.4, "aceptable", A),
                (0.4, 0.6, "moderado", A), (0.6, 0.8, "sustancial", G), (0.8, 1, "casi perfecto", V)],
        fuente="Landis y Koch (1977)"),
    "auc": dict(
        nombre="AUC uno-contra-resto (macro)", grupo="Desempeño",
        mide="Probabilidad de que el modelo dé más probabilidad de un estrato a una vivienda de ese estrato que a una "
             "que no lo es. Se promedia entre estratos. Mide la capacidad de ordenar, no la clase final.",
        rango="0.5 (azar) a 1", mejor="alto",
        tramos=[(0.5, 0.7, "pobre", R), (0.7, 0.8, "aceptable", A), (0.8, 0.9, "excelente", G), (0.9, 1, "sobresaliente", V)],
        fuente="Hosmer, Lemeshow y Sturdivant (2013)"),
    "log_loss": dict(
        nombre="Log loss (entropía cruzada)", grupo="Desempeño",
        mide="Penaliza las probabilidades mal asignadas: castiga mucho dar probabilidad casi 0 al estrato real.",
        rango="0 (perfecto) a ∞; dar 1/6 a cada estrato da ln 6 ≈ 1.79", mejor="bajo",
        tramos=[(0, 0.7, "probabilidades buenas", V), (0.7, 1.2, "aceptables", G), (1.2, 1.79, "poco informativas", A),
                (1.79, 3, "peor que repartir por igual", R)]),
    "brier": dict(
        nombre="Brier multiclase", grupo="Desempeño",
        mide="Error cuadrático medio entre el vector de probabilidades y el estrato real (codificado como 0/1).",
        rango="0 (perfecto) a 2; repartir 1/6 a cada estrato da 0.83", mejor="bajo",
        tramos=[(0, 0.4, "bueno", V), (0.4, 0.6, "aceptable", G), (0.6, 0.83, "débil", A), (0.83, 2, "peor que repartir", R)]),
    "ece": dict(
        nombre="Error de calibración esperado (ECE)", grupo="Desempeño",
        mide="Diferencia promedio entre la probabilidad predicha y la frecuencia observada. Si el modelo dice 70 %, ¿acierta el 70 % de las veces?",
        rango="0 (perfectamente calibrado) a 1", mejor="bajo",
        tramos=[(0, 0.05, "bien calibrado", V), (0.05, 0.1, "aceptable", G), (0.1, 0.2, "descalibrado", A), (0.2, 1, "muy descalibrado", R)]),
    # ------------------------------------------------------------------ validación e incertidumbre
    "ic95": dict(
        nombre="Intervalo de confianza del 95 % (bootstrap por bloques)", grupo="Validación",
        mide="Rango plausible de una métrica o de una diferencia entre modelos. Se obtiene remuestreando bloques de 2 km "
             "(no viviendas), porque las viviendas de un bloque no son independientes.",
        rango="—", mejor="",
        tramos=[],
        nota="Para una diferencia entre modelos: si el intervalo excluye el 0, la diferencia es significativa; si lo "
             "incluye, no se puede distinguir del ruido. Un intervalo ancho indica pocos bloques, no un mal modelo."),
    "se_pareado": dict(
        nombre="Error estándar pareado y criterio 1-SE", grupo="Validación",
        mide="Desviación de las diferencias fold a fold entre dos modelos, dividida por √(número de folds). Un modelo "
             "reemplaza a la logística solo si su ventaja media supera un error estándar.",
        rango="≥ 0", mejor="",
        tramos=[],
        nota="Compara a los dos modelos en las mismas zonas, así que descuenta que unos folds sean más difíciles que otros."),
    "p_valor": dict(
        nombre="p-valor", grupo="Validación",
        mide="Probabilidad de ver un resultado al menos tan extremo si no hubiera efecto.",
        rango="0 a 1", mejor="bajo",
        tramos=[(0, 0.01, "muy significativo", V), (0.01, 0.05, "significativo", G), (0.05, 0.1, "marginal", A), (0.1, 1, "no significativo", R)],
        nota="Con cientos de miles de viviendas casi todo es significativo: lo que importa es el tamaño del efecto "
             "(η², V de Cramér, diferencia de F1). Con muchas pruebas a la vez se corrige con Holm o Benjamini-Hochberg."),
    "buffer": dict(
        nombre="Buffer espacial (km)", grupo="Validación",
        mide="Franja alrededor de cada zona de validación o prueba cuyas viviendas se quitan del entrenamiento, para "
             "que el modelo no aprenda de vecinos casi idénticos.",
        rango="0 km en adelante", mejor="",
        tramos=[(0, 0.5, "insuficiente: fuga por cercanía", R), (0.5, 2.6, "parcial (el usado: 1 km)", A),
                (2.6, 5, "≥ alcance del correlograma: sin fuga", V)],
        nota="Más buffer quita más fuga, pero también más datos: con 2.6 km el F1 de CV cae a 0.15 porque se pierde "
             "demasiado entrenamiento. 1 km fue el compromiso elegido."),
    "optimismo": dict(
        nombre="Optimismo de la validación aleatoria", grupo="Validación",
        mide="Cuánto más alto sale el F1 si se mezclan filas al azar en vez de separar zonas.",
        rango="≥ 0", mejor="bajo",
        tramos=[(0, 0.05, "despreciable", V), (0.05, 0.15, "moderado", A), (0.15, 1, "grave: la validación aleatoria engaña", R)]),
    # ------------------------------------------------------------------ espacial
    "moran": dict(
        nombre="I de Moran global", grupo="Espacial",
        mide="Autocorrelación espacial: si las viviendas vecinas (aquí, los 8 edificios más cercanos) tienen valores "
             "parecidos. Es una correlación entre el valor de cada punto y el promedio de sus vecinos.",
        rango="aprox. −1 (vecinos opuestos) a 1 (vecinos iguales); 0 = sin patrón espacial", mejor="",
        tramos=[(-1, -0.3, "dispersión (vecinos distintos)", A), (-0.3, 0.1, "sin patrón", GR), (0.1, 0.3, "débil", "#9ec5f4"),
                (0.3, 0.7, "moderada", G), (0.7, 1, "fuerte", "#0d366b")],
        nota="Los cortes son orientativos; la significancia se juzga con un p de permutación. En los residuos de un "
             "modelo, el ideal es 0: un valor alto dice que al modelo le falta información de barrio.",
        fuente="Moran (1950); Anselin (1995)"),
    "lisa": dict(
        nombre="LISA (Moran local)", grupo="Espacial",
        mide="Descompone el I de Moran por edificio. Clasifica cada punto: HH (alto rodeado de altos), LL (bajo entre "
             "bajos), HL y LH (atípicos espaciales) o no significativo.",
        rango="categorías", mejor="", tramos=[],
        nota="HH y LL son barrios homogéneos; HL y LH son las viviendas que más le cuesta predecir a un modelo basado en la ubicación."),
    "gi": dict(
        nombre="Getis-Ord Gi*", grupo="Espacial",
        mide="Puntaje z de la suma del estrato en el vecindario (incluido el propio punto): puntos calientes y fríos.",
        rango="z estándar", mejor="",
        tramos=[(-6, -1.96, "punto frío significativo", "#2c7bb6"), (-1.96, 1.96, "no significativo", GR),
                (1.96, 6, "punto caliente significativo", R)]),
    "correlograma": dict(
        nombre="Correlograma y alcance", grupo="Espacial",
        mide="El I de Moran calculado por bandas de distancia. El alcance es la distancia a la que el parecido entre "
             "vecinos se vuelve débil (I < 0.3).",
        rango="km", mejor="", tramos=[],
        nota="Aquí el alcance es 2.6 km: dos viviendas a menos de esa distancia siguen pareciéndose."),
    "clark_evans": dict(
        nombre="Índice de Clark-Evans (R)", grupo="Espacial",
        mide="Distancia media al vecino más cercano frente a la esperada si los puntos estuvieran al azar.",
        rango="0 (todo en un punto) a 2.15 (rejilla perfecta)", mejor="",
        tramos=[(0, 0.9, "agrupado", G), (0.9, 1.1, "aleatorio", GR), (1.1, 2.15, "regular", A)]),
    "icc": dict(
        nombre="ICC(1) (correlación intraclase)", grupo="Datos",
        mide="Qué parte de la variación del estrato está entre grupos (predios matriz) y no dentro de ellos.",
        rango="0 (grupos no se parecen por dentro) a 1 (todo el grupo tiene el mismo estrato)", mejor="",
        tramos=[(0, 0.5, "dependencia baja", V), (0.5, 0.75, "moderada", A), (0.75, 0.9, "alta", R), (0.9, 1, "casi total", R)],
        nota="Aquí no es una métrica de calidad: un ICC alto (0.992) obliga a no partir edificios entre train y test.",
        fuente="Koo y Li (2016), escala de confiabilidad"),
    "deff": dict(
        nombre="Efecto de diseño de Kish y tamaño efectivo", grupo="Datos",
        mide="deff = 1 + (m̃ − 1)·ICC: cuánto se inflan las varianzas por tener filas agrupadas. Tamaño efectivo = N / deff.",
        rango="deff ≥ 1 (1 = filas independientes)", mejor="bajo",
        tramos=[(1, 2, "dependencia despreciable", V), (2, 10, "importante", A), (10, 1000, "las filas valen mucho menos que su número", R)]),
    # ------------------------------------------------------------------ EDA: univariado y bivariado
    "asimetria": dict(
        nombre="Asimetría (skewness)", grupo="EDA",
        mide="Qué tan desigual es la distribución a cada lado de la media. Positiva: cola larga a la derecha.",
        rango="−∞ a ∞ (0 = simétrica)", mejor="",
        tramos=[(0, 0.5, "aprox. simétrica", V), (0.5, 1, "moderada", A), (1, 50, "alta: transformar (log)", R)],
        nota="Se lee en valor absoluto.", fuente="Bulmer (1979)"),
    "curtosis": dict(
        nombre="Curtosis (en exceso)", grupo="EDA",
        mide="Peso de las colas frente a una normal (0). Positiva: más valores extremos de lo normal.",
        rango="≥ −2", mejor="",
        tramos=[(0, 0.5, "similar a la normal", V), (0.5, 1, "moderada", A), (1, 500, "colas pesadas", R)],
        nota="Se lee en valor absoluto; colas pesadas justifican winsorizar y usar mediana e IQR."),
    "normalidad": dict(
        nombre="Pruebas de normalidad (D'Agostino-Pearson, Shapiro-Wilk)", grupo="EDA",
        mide="Si la distribución es compatible con una normal. Hipótesis nula: es normal.",
        rango="p de 0 a 1", mejor="",
        tramos=[(0, 0.05, "se rechaza la normalidad", R), (0.05, 1, "compatible con normal", V)],
        nota="Con muestras grandes rechazan casi siempre; sirven como confirmación, la decisión se toma con la asimetría."),
    "iqr": dict(
        nombre="Regla de Tukey (IQR)", grupo="EDA",
        mide="Marca como atípico lo que está a más de 1.5 (moderado) o 3 (extremo) rangos intercuartílicos fuera de Q1 o Q3.",
        rango="% de filas", mejor="",
        tramos=[(0, 1, "pocos", V), (1, 5, "habitual en datos asimétricos", A), (5, 100, "muchos: la variable es muy asimétrica", R)],
        nota="En distribuciones asimétricas marca muchos valores legítimos: aquí se marcan, no se borran."),
    "spearman": dict(
        nombre="Correlación de Spearman (ρ) y de Pearson (r)", grupo="EDA",
        mide="Pearson mide relación lineal; Spearman, relación monótona (Pearson sobre rangos), robusta a colas y outliers.",
        rango="−1 a 1 (0 = sin relación; el signo da la dirección)", mejor="",
        tramos=[(0, 0.1, "despreciable", GR), (0.1, 0.3, "débil", "#9ec5f4"), (0.3, 0.5, "moderada", G), (0.5, 0.7, "fuerte", "#1c5cab"),
                (0.7, 1, "muy fuerte", "#0d366b")],
        nota="Se lee en valor absoluto. Si |r − ρ| ≥ 0.2, la relación no es lineal o hay outliers: se prioriza Spearman."),
    "eta2": dict(
        nombre="η² de Kruskal-Wallis", grupo="EDA",
        mide="Tamaño de efecto: fracción de la variación (en rangos) de una variable numérica que se explica por el estrato.",
        rango="0 a 1", mejor="",
        tramos=[(0, 0.01, "trivial", GR), (0.01, 0.06, "pequeño", "#9ec5f4"), (0.06, 0.14, "mediano", G), (0.14, 1, "grande", "#0d366b")],
        fuente="Cohen (1988)"),
    "cramer": dict(
        nombre="V de Cramér (y prueba χ²)", grupo="EDA",
        mide="Fuerza de la asociación entre dos variables categóricas (por ejemplo, régimen de propiedad y estrato). "
             "La prueba χ² dice si hay asociación; la V dice cuánta.",
        rango="0 (independencia) a 1 (asociación perfecta)", mejor="",
        tramos=[(0, 0.1, "despreciable", GR), (0.1, 0.3, "débil", "#9ec5f4"), (0.3, 0.5, "moderada", G), (0.5, 1, "fuerte", "#0d366b")],
        nota="Cortes de Cohen para tablas 2×k; con 6 estratos son conservadores.", fuente="Cohen (1988)"),
    "mi": dict(
        nombre="Información mutua normalizada (MI / H(Y))", grupo="EDA",
        mide="Qué fracción de la incertidumbre sobre el estrato elimina conocer una variable. Capta relaciones no "
             "lineales y sirve igual para numéricas y categóricas.",
        rango="0 (no informa) a 1 (determina el estrato)", mejor="",
        tramos=[(0, 0.05, "poca información", GR), (0.05, 0.15, "algo", "#9ec5f4"), (0.15, 0.3, "bastante", G), (0.3, 1, "mucha", "#0d366b")],
        nota="Cortes orientativos del proyecto: sirve sobre todo para ordenar variables."),
    "vif": dict(
        nombre="VIF (factor de inflación de la varianza)", grupo="EDA",
        mide="Cuánto se explica una variable con las demás: VIF = 1 / (1 − R²). √VIF es cuánto se infla el error "
             "estándar de su coeficiente.",
        rango="≥ 1", mejor="bajo",
        tramos=[(1, 5, "aceptable", V), (5, 10, "moderada: vigilar", A), (10, 1000, "multicolinealidad alta", R)]),
    "auc_univariado": dict(
        nombre="AUC de una sola variable (auditoría de fuga)", grupo="EDA",
        mide="Qué tan bien separa los estratos una variable sola. Un valor casi perfecto suele delatar una variable "
             "derivada del objetivo (fuga).",
        rango="0.5 (azar) a 1", mejor="",
        tramos=[(0.5, 0.6, "casi nulo", GR), (0.6, 0.7, "débil", "#9ec5f4"), (0.7, 0.85, "informativo", G),
                (0.85, 0.95, "muy alto: revisar", A), (0.95, 1, "alerta de fuga", R)]),
    "smd": dict(
        nombre="Diferencia de medias estandarizada (SMD)", grupo="EDA",
        mide="Diferencia entre dos grupos (por ejemplo, filas con y sin dato) en unidades de desviación estándar.",
        rango="≥ 0 (en valor absoluto)", mejor="",
        tramos=[(0, 0.1, "despreciable", V), (0.1, 0.2, "pequeña", A), (0.2, 5, "relevante: el faltante no es al azar", R)],
        fuente="Austin (2009)"),
    # ------------------------------------------------------------------ multivariado
    "varianza_pca": dict(
        nombre="Varianza explicada (PCA)", grupo="Multivariado",
        mide="Qué parte de la variación conjunta de las variables resume cada componente principal.",
        rango="0 a 1 por componente; suma 1", mejor="",
        tramos=[], nota="Criterios habituales: quedarse con los componentes hasta el 80 % acumulado, o con los de autovalor > 1 (Kaiser)."),
    "participation_ratio": dict(
        nombre="Dimensionalidad efectiva (participation ratio)", grupo="Multivariado",
        mide="(Σλ)² / Σλ²: cuántas dimensiones \"de verdad\" tienen los datos.",
        rango="1 (todo en una dirección) a p (todas las variables independientes)", mejor="", tramos=[]),
    "cargas": dict(
        nombre="Cargas (loadings) del PCA", grupo="Multivariado",
        mide="Peso de cada variable en un componente. El signo da la dirección y el valor absoluto, la importancia.",
        rango="−1 a 1", mejor="",
        tramos=[(0, 0.2, "irrelevante", GR), (0.2, 0.4, "aporta", "#9ec5f4"), (0.4, 1, "define el componente", "#0d366b")],
        nota="Se lee en valor absoluto."),
    "silueta": dict(
        nombre="Coeficiente de silueta", grupo="Multivariado",
        mide="Qué tan cerca está cada punto de su grupo frente al grupo vecino. Mide si los grupos de un clustering existen de verdad.",
        rango="−1 a 1", mejor="alto",
        tramos=[(-1, 0.25, "sin estructura", R), (0.25, 0.5, "débil, posiblemente artificial", A), (0.5, 0.7, "razonable", G),
                (0.7, 1, "fuerte", V)],
        fuente="Kaufman y Rousseeuw (1990)"),
    "mahalanobis": dict(
        nombre="Distancia de Mahalanobis robusta (MCD) e Isolation Forest", grupo="Multivariado",
        mide="Detectan combinaciones raras de variables. Mahalanobis: distancia al centro teniendo en cuenta las "
             "correlaciones; se marca lo que supera el percentil 99.9 de una χ². Isolation Forest: puntos fáciles de aislar con cortes al azar.",
        rango="% de filas marcadas", mejor="", tramos=[],
        nota="Si ambos métodos coinciden (Jaccard alto), el atípico es más creíble."),
    "jaccard": dict(
        nombre="Índice de Jaccard", grupo="Multivariado",
        mide="Coincidencia entre dos conjuntos: intersección / unión.",
        rango="0 (nada en común) a 1 (idénticos)", mejor="alto",
        tramos=[(0, 0.2, "poca coincidencia", R), (0.2, 0.5, "parcial", A), (0.5, 1, "alta", V)]),
    # ------------------------------------------------------------------ modelos e hiperparámetros
    "C": dict(
        nombre="C (logística)", grupo="Hiperparámetros",
        mide="Inverso de la fuerza de la regularización L2. C pequeño encoge los coeficientes (modelo más simple); C grande los deja libres.",
        rango="> 0 (se probaron 0.01, 0.1, 1 y 10)", mejor="", tramos=[],
        nota="No hay un valor bueno a priori: se elige con la CV espacial y la regla 1-SE (el más simple dentro de un error estándar del mejor)."),
    "class_weight": dict(
        nombre="Pesos de clase (class_weight)", grupo="Hiperparámetros",
        mide="'balanced' pondera cada vivienda por 1 / frecuencia de su estrato, para que los estratos pequeños pesen lo mismo al entrenar.",
        rango="sin pesos o balanceados", mejor="", tramos=[],
        nota="Sube el recall de las clases pequeñas a costa de algo de accuracy."),
    "odds_ratio": dict(
        nombre="Odds ratio", grupo="Modelos",
        mide="Por cuánto se multiplican las odds de un estrato (frente al 1) al subir una variable una desviación estándar.",
        rango="> 0 (1 = sin efecto)", mejor="",
        tramos=[(0, 0.5, "baja fuertemente las odds", R), (0.5, 0.9, "las baja", A), (0.9, 1.1, "sin efecto", GR),
                (1.1, 2, "las sube", G), (2, 100, "las sube fuertemente", "#0d366b")]),
    "importancia_perm": dict(
        nombre="Importancia por permutación", grupo="Modelos",
        mide="Cuánto cae el F1 si se barajan los valores de una variable (se rompe su relación con el estrato).",
        rango="≥ 0 (puede ser levemente negativa por azar)", mejor="", tramos=[],
        nota="Una caída cercana a 0 dice que el modelo puede prescindir de la variable."),
    "alpha_geo": dict(
        nombre="α del XGBoost geográfico", grupo="Hiperparámetros",
        mide="Peso del modelo global en la mezcla: p = α·global + (1 − α)·local. α = 1 es el XGBoost normal; α = 0 usa solo los modelos locales.",
        rango="0 a 1", mejor="", tramos=[]),
    "gamma_rbf": dict(
        nombre="γ del kernel RBF (SVM)", grupo="Hiperparámetros",
        mide="Qué tan local es el kernel: γ grande = cada vivienda solo se parece a sus vecinas muy cercanas (fronteras más irregulares).",
        rango="> 0", mejor="", tramos=[]),
}

GRUPOS = ["Desempeño", "Validación", "Espacial", "Datos", "EDA", "Multivariado", "Modelos", "Hiperparámetros"]
