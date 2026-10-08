"""Genera las imágenes del dashboard a partir de los datos (se ejecuta una vez; las salidas quedan en assets/).

Autores: María Carolina Cantillo Orozco (200179105) y
         Juan Camilo Oñoro Araujo (200177329)

- assets/portada.svg: "Barranquilla de noche". Cada luz es una zona de unos 165 m con viviendas; el tamaño
  crece con el número de viviendas y el brillo con el estrato medio (más brillante = estrato más alto).
- assets/escala_estratos.svg: infografía de la escala de estratos y su efecto en las tarifas (Ley 142 de 1994).

Uso:  python generar_portada.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "assets"

# Escala luminosa para fondo oscuro: estrato bajo = luz tenue y fría, estrato alto = luz brillante y cálida.
LUZ = ["#2c5d9e", "#3f86d4", "#5fb4ea", "#a9dcf2", "#fde7a6", "#ffc75a"]


def _mezcla(c1, c2, t):
    a = np.array([int(c1[i:i + 2], 16) for i in (1, 3, 5)])
    b = np.array([int(c2[i:i + 2], 16) for i in (1, 3, 5)])
    r = (a + (b - a) * t).round().astype(int)
    return "#" + "".join(f"{v:02x}" for v in r)


def color_estrato(e):
    e = float(np.clip(e, 1, 6))
    i = min(int(np.floor(e)) - 1, 4)
    return _mezcla(LUZ[i], LUZ[i + 1], e - (i + 1))


def portada(celda_grados=0.0015, ancho=1200, alto=760):
    v = pd.read_pickle(AQUI / "datos" / "viviendas.pkl.gz")
    v = v[v["tiene_coordenadas"]].dropna(subset=["lat", "lon"])
    lat0 = float(v["lat"].mean())
    # Proyección local en km (equirectangular), igual idea que el Entregable 1.
    x = (v["lon"].astype(float) - v["lon"].astype(float).min()) * 111.32 * np.cos(np.radians(lat0))
    y = (v["lat"].astype(float) - v["lat"].astype(float).min()) * 110.57
    g = pd.DataFrame({"gx": np.floor(x / (celda_grados * 111)), "gy": np.floor(y / (celda_grados * 111)),
                      "x": x, "y": y, "e": v["estrato_num"].astype(float)})
    z = g.groupby(["gx", "gy"]).agg(x=("x", "mean"), y=("y", "mean"), e=("e", "mean"), n=("e", "size")).reset_index()
    xmax, ymax = float(x.max()), float(y.max())
    margen = 40
    esc = min((ancho - 2 * margen) / xmax, (alto - 2 * margen) / ymax)
    dx = (ancho - xmax * esc) / 2
    dy = (alto - ymax * esc) / 2
    z = z.sort_values("e")  # las luces más brillantes quedan encima
    circulos = []
    for r in z.itertuples():
        cx = dx + r.x * esc
        cy = alto - (dy + r.y * esc)
        radio = 1.1 + 0.55 * np.log1p(r.n)
        op = 0.55 + 0.45 * (r.e - 1) / 5
        circulos.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radio:.2f}" fill="{color_estrato(r.e)}" '
                        f'fill-opacity="{op:.2f}"/>')
    km = esc  # píxeles por km
    leyenda = "".join(
        f'<circle cx="{ancho - 250 + i * 38}" cy="{alto - 46}" r="8" fill="{LUZ[i]}" filter="url(#brillo)"/>'
        f'<text x="{ancho - 250 + i * 38}" y="{alto - 20}" text-anchor="middle" class="t">{i + 1}</text>'
        for i in range(6))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ancho} {alto}" role="img"
 aria-label="Mapa de Barranquilla: cada luz es una zona con viviendas; el brillo indica el estrato medio">
<title>Barranquilla de noche: el estrato visto desde el catastro</title>
<defs>
 <radialGradient id="cielo" cx="55%" cy="40%" r="75%">
  <stop offset="0" stop-color="#0d2a52"/><stop offset="0.6" stop-color="#071a36"/><stop offset="1" stop-color="#030c1c"/>
 </radialGradient>
 <filter id="brillo" x="-50%" y="-50%" width="200%" height="200%">
  <feGaussianBlur stdDeviation="2.2" result="b"/>
  <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
 </filter>
 <style>.t{{font:600 15px Inter,Segoe UI,Arial,sans-serif;fill:#c9d8ee}} .s{{font:400 13px Inter,Segoe UI,Arial,sans-serif;fill:#8fa7c9}}</style>
</defs>
<rect width="{ancho}" height="{alto}" fill="url(#cielo)"/>
<g filter="url(#brillo)">{"".join(circulos)}</g>
<g transform="translate(46,60)"><path d="M0 26 L10 0 L20 26 L10 20 Z" fill="#c9d8ee"/><text x="10" y="46" text-anchor="middle" class="t">N</text></g>
<g transform="translate(40,{alto - 40})"><rect width="{2 * km:.1f}" height="4" fill="#c9d8ee"/><text x="0" y="-8" class="s">2 km</text></g>
<text x="{ancho - 40}" y="{alto - 74}" text-anchor="end" class="s">estrato medio de la zona (más brillante = más alto)</text>
{leyenda}
</svg>'''
    (SALIDA / "portada.svg").write_text(svg, encoding="utf-8")
    return len(circulos)


def escala_estratos(ancho=1100, alto=330):
    """Seis viviendas que crecen con el estrato y la regla tarifaria de la Ley 142 de 1994."""
    azul = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
    nombres = ["Bajo-bajo", "Bajo", "Medio-bajo", "Medio", "Medio-alto", "Alto"]
    base = 210
    piezas = []
    paso = ancho / 6
    for i in range(6):
        cx = paso * i + paso / 2
        w = 52 + i * 12
        h = 46 + i * 16
        x0 = cx - w / 2
        techo = f"{x0 - 8:.0f},{base - h:.0f} {cx:.0f},{base - h - 30 - i * 3:.0f} {x0 + w + 8:.0f},{base - h:.0f}"
        piezas.append(
            f'<polygon points="{techo}" fill="{azul[i]}" opacity="0.85"/>'
            f'<rect x="{x0:.0f}" y="{base - h:.0f}" width="{w}" height="{h}" rx="3" fill="{azul[i]}"/>'
            f'<rect x="{cx - 7:.0f}" y="{base - 26}" width="14" height="26" fill="#ffffff" opacity="0.85"/>'
            + "".join(f'<rect x="{x0 + 8 + k * 16:.0f}" y="{base - h + 10:.0f}" width="9" height="9" fill="#ffffff" '
                      f'opacity="0.7"/>' for k in range(max(1, (w - 12) // 16)))
            + f'<text x="{cx:.0f}" y="{base + 28}" text-anchor="middle" class="n">{i + 1}</text>'
            f'<text x="{cx:.0f}" y="{base + 48}" text-anchor="middle" class="e">{nombres[i]}</text>')

    def llave(x1, x2, color, texto):
        return (f'<path d="M{x1 + 10:.0f} {base + 66} L{x1 + 10:.0f} {base + 76} L{x2 - 10:.0f} {base + 76} '
                f'L{x2 - 10:.0f} {base + 66}" fill="none" stroke="{color}" stroke-width="3"/>'
                f'<text x="{(x1 + x2) / 2:.0f}" y="{base + 100}" text-anchor="middle" class="r" fill="{color}">{texto}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 25 {ancho} {alto - 25}" role="img"
 aria-label="Escala de estratos 1 a 6: los estratos 1 a 3 reciben subsidio, el 4 paga la tarifa plena y los 5 y 6 pagan contribución">
<title>Escala de estratos y tarifas de servicios públicos</title>
<style>.n{{font:700 22px Inter,Segoe UI,Arial,sans-serif;fill:#0b0b0b}} .e{{font:400 15px Inter,Segoe UI,Arial,sans-serif;fill:#52514e}}
.r{{font:600 16px Inter,Segoe UI,Arial,sans-serif}}</style>
<line x1="20" y1="{base}" x2="{ancho - 20}" y2="{base}" stroke="#c3c2b7" stroke-width="2"/>
{"".join(piezas)}
{llave(0, paso * 3, "#1baf7a", "Subsidio en la tarifa")}
{llave(paso * 3, paso * 4, "#52514e", "Tarifa plena")}
{llave(paso * 4, paso * 6, "#eb6834", "Contribución adicional")}
</svg>'''
    (SALIDA / "escala_estratos.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    n = portada()
    escala_estratos()
    print(f"portada.svg con {n} zonas; escala_estratos.svg generada")
