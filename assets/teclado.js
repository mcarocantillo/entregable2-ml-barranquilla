// Navegación con el teclado: flechas izquierda/derecha cambian de diapositiva (como en una presentación).
// En la vista continua saltan a la diapositiva siguiente o anterior de la página, y el menú marca la que se ve.
function diapoActual(secciones) {
  let actual = 0;
  secciones.forEach(function (s, i) { if (s.getBoundingClientRect().top <= 120) { actual = i; } });
  return actual;
}
document.addEventListener("keydown", function (e) {
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) { return; }
  if (t && t.closest && t.closest(".Select, .dash-dropdown, .rc-slider")) { return; }
  const paso = e.key === "ArrowRight" ? 1 : (e.key === "ArrowLeft" ? -1 : 0);
  if (!paso) { return; }
  const secciones = Array.from(document.querySelectorAll(".diapo-continua"));
  if (secciones.length) {
    const j = Math.min(Math.max(diapoActual(secciones) + paso, 0), secciones.length - 1);
    secciones[j].scrollIntoView({ behavior: "smooth", block: "start" });
    e.preventDefault();
    return;
  }
  const b = document.getElementById(paso > 0 ? "btn-siguiente" : "btn-anterior");
  if (b && !b.disabled) { b.click(); e.preventDefault(); }
});
document.addEventListener("scroll", function () {
  const secciones = Array.from(document.querySelectorAll(".diapo-continua"));
  if (!secciones.length) { return; }
  const i = diapoActual(secciones);
  document.querySelectorAll(".menu-enlace").forEach(function (a, j) { a.classList.toggle("activo", j === i); });
}, { passive: true });
