/* Mide la barra de navegación fija y guarda su alto en la variable CSS --alto-nav.
   La usan el menú lateral (sticky), los saltos de la vista continua y la altura de las gráficas, para que la
   diapositiva quepa en la pantalla aunque la barra ocupe una o dos filas según el ancho. */
(function () {
  function medir() {
    var nav = document.getElementById('navegacion');
    if (!nav) { return false; }
    var fijar = function () {
      document.documentElement.style.setProperty('--alto-nav', Math.ceil(nav.getBoundingClientRect().height) + 'px');
    };
    fijar();
    if (window.ResizeObserver) { new ResizeObserver(fijar).observe(nav); } else { window.addEventListener('resize', fijar); }
    return true;
  }
  var intentos = 0;
  var esperar = setInterval(function () { if (medir() || ++intentos > 100) { clearInterval(esperar); } }, 100);
})();
