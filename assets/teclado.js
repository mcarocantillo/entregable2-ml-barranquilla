// Navegación con el teclado: flechas izquierda/derecha cambian de diapositiva (como en una presentación).
document.addEventListener("keydown", function (e) {
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) { return; }
  if (t && t.closest && t.closest(".Select, .dash-dropdown, .rc-slider")) { return; }
  const id = e.key === "ArrowRight" ? "btn-siguiente" : (e.key === "ArrowLeft" ? "btn-anterior" : null);
  if (!id) { return; }
  const b = document.getElementById(id);
  if (b && !b.disabled) { b.click(); e.preventDefault(); }
});
