"""Variables de vecindad física para el estrato (Entregable 2).

Autores: María Carolina Cantillo Orozco (200179105) y
         Juan Camilo Oñoro Araujo (200177329)

Para cada vivienda se resume cómo son las viviendas que la rodean, a dos escalas
(300 m y 1 km): tamaño medio, baños, antigüedad, proporción de apartamentos en
edificios altos, proporción de predios informales y densidad.

Reglas que evitan la fuga de datos:
- Solo se usan atributos FÍSICOS del catastro. Nunca el estrato de los vecinos
  (en una zona nueva no se conoce), así que estas variables están disponibles
  tanto en entrenamiento como en zonas sin datos.
- Se excluyen las demás viviendas del MISMO edificio: sus atributos ya los ve el
  modelo en la propia vivienda y repetirlos solo copiaría información.
- No se aprende nada de los datos (no hay medias ni cuantiles de entrenamiento):
  es una regla geométrica fija. Los pasos que sí aprenden (imputación, escalado)
  van en el Pipeline y se ajustan solo con el entrenamiento de cada fold.

Cálculo: las viviendas se agrupan en una rejilla de 50 m sobre las coordenadas
proyectadas (km) y la suma de cada atributo dentro de un disco de radio r se
obtiene con una convolución, lo que es exacto salvo por el redondeo de la
posición a la celda (≤ 50 m, pequeño frente a 300 m y 1 km).
"""
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

RADIOS_KM = (0.3, 1.0)
CELDA_KM = 0.05


def _disco(radio_celdas):
    r = int(np.ceil(radio_celdas - 1e-9))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    # la tolerancia evita que 0.3/0.05 = 5.999... deje fuera las celdas a exactamente r
    return (xx ** 2 + yy ** 2 <= radio_celdas ** 2 + 1e-6).astype(float)


def _suma_en_disco(valores_por_celda, disco):
    """Suma de cada capa dentro del disco centrado en cada celda (convolución)."""
    out = np.stack([fftconvolve(capa, disco, mode="same") for capa in valores_por_celda])
    return np.clip(out, 0.0, None)  # el redondeo de la FFT puede dejar ~-1e-12


def calcular_vecindad(df, radios_km=RADIOS_KM, celda_km=CELDA_KM, col_edificio="npn_edificio", devolver_diagnostico=False):
    """Devuelve un DataFrame con las variables de vecindad, en el mismo orden de ``df``.

    Parámetros
    ----------
    df : DataFrame con x_km, y_km (sin NaN), el identificador de edificio y los atributos
         area_construida, total_banios, antiguedad, uso y condicion_predio.
    radios_km : radios de las vecindades, en km.
    celda_km : lado de la rejilla de acumulación.
    """
    x = df["x_km"].to_numpy(float)
    y = df["y_km"].to_numpy(float)
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("calcular_vecindad requiere coordenadas sin NaN (use solo filas con coordenadas)")
    n = len(df)
    pad = int(np.ceil(max(radios_km) / celda_km)) + 2
    cx = np.floor((x - x.min()) / celda_km).astype(int) + pad
    cy = np.floor((y - y.min()) / celda_km).astype(int) + pad
    forma = (cy.max() + pad + 1, cx.max() + pad + 1)

    # Atributos físicos por vivienda: cada uno aporta una suma y un conteo de valores válidos.
    area = np.log1p(df["area_construida"].to_numpy(float))
    ban = df["total_banios"].to_numpy(float)
    ant = df["antiguedad"].to_numpy(float)
    apto = df["uso"].astype(str).str.contains("Apartamentos", case=False).to_numpy(float)
    inf = (df["condicion_predio"].astype(str) == "Informal").to_numpy(float)
    capas = {
        "unidades": np.ones(n),
        "area_sum": np.nan_to_num(area), "area_n": (~np.isnan(area)).astype(float),
        "ban_sum": np.nan_to_num(ban), "ban_n": (~np.isnan(ban)).astype(float),
        "ant_sum": np.nan_to_num(ant), "ant_n": (~np.isnan(ant)).astype(float),
        "apto_sum": apto, "inf_sum": inf,
    }
    nombres = list(capas)

    # Rejilla de acumulación (una capa por atributo) y totales por edificio, para excluirlo.
    rejilla = []
    for k in nombres:
        g = np.zeros(forma)
        np.add.at(g, (cy, cx), capas[k])
        rejilla.append(g)
    edif = df[col_edificio].astype(str).to_numpy()
    tot_edif = pd.DataFrame({k: capas[k] for k in nombres}).groupby(edif).transform("sum")

    salida = {}
    diag = {}
    for r in radios_km:
        disco = _disco(r / celda_km)
        S = _suma_en_disco(rejilla, disco)                    # (capas, filas, columnas)
        vec = {k: S[i][cy, cx] for i, k in enumerate(nombres)}  # suma en la vecindad de cada vivienda
        crudo_neg = 0
        for k in nombres:                                      # se resta el propio edificio
            resta = vec[k] - tot_edif[k].to_numpy()
            crudo_neg += int((resta < -0.5).sum())
            vec[k] = np.clip(resta, 0.0, None)
        etq = f"vec{int(round(r * 1000))}"
        nv = vec["unidades"]
        with np.errstate(invalid="ignore", divide="ignore"):
            salida[f"{etq}_area_log"] = np.where(vec["area_n"] > 0.5, vec["area_sum"] / vec["area_n"], np.nan)
            salida[f"{etq}_banios"] = np.where(vec["ban_n"] > 0.5, vec["ban_sum"] / vec["ban_n"], np.nan)
            salida[f"{etq}_antiguedad"] = np.where(vec["ant_n"] > 0.5, vec["ant_sum"] / vec["ant_n"], np.nan)
            salida[f"{etq}_apto"] = np.where(nv > 0.5, vec["apto_sum"] / nv, np.nan)
            salida[f"{etq}_informal"] = np.where(nv > 0.5, vec["inf_sum"] / nv, np.nan)
        salida[f"{etq}_dens"] = np.log1p(nv)
        diag[etq] = {"vecindades_vacias": float((nv < 0.5).mean()), "resta_negativa_recortada": crudo_neg,
                     "unidades_vecinas_mediana": float(np.median(nv))}
    out = pd.DataFrame(salida, index=df.index)
    return (out, diag) if devolver_diagnostico else out
