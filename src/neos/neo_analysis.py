import warnings
import astropy.units as u
import lightkurve as lk
import numpy as np
import pandas as pd

# Evitamos avisos molestos de lightkurve que no afectan el cálculo
warnings.filterwarnings("ignore", category=lk.LightkurveWarning)


def _arr(x):
    """Convierte cantidades de astropy/lightkurve o listas a arreglo float limpio."""
    return np.asarray(getattr(x, "value", x), dtype=float)


def _val(x):
    """Extrae un escalar float de un objeto con unidades o valor numérico."""
    return float(getattr(x, "value", x))


def search_neo_lightcurves(target, mission=None):
    """
    Busca en MAST las curvas de luz disponibles para un asteroide o NEO.
    Si se indica 'mission' (por ejemplo 'K2' o 'TESS'), filtra por esa misión.
    """
    # Revisamos que el nombre o designación no venga vacío
    if not target or not str(target).strip():
        raise ValueError("El nombre o designación del objeto no puede estar vacío.")

    query = str(target).strip()
    if mission:
        return lk.search_lightcurve(query, mission=mission)
    return lk.search_lightcurve(query)


def prepare_neo_lightcurve(lc):
    """
    Limpia valores nulos o no finitos de una curva de luz y ordena en tiempo.
    Regresa arreglos listos para analizar: tiempo (t), flujo (f) y error (ferr).
    """
    # Quitamos NaNs antes de sacar arreglos numéricos
    clean = lc.remove_nans()
    t = _arr(clean.time.value)
    f = _arr(clean.flux)

    # Si la curva no trae columna de error, rellenamos con NaN para mantener la forma
    if getattr(clean, "flux_err", None) is not None:
        ferr = _arr(clean.flux_err)
    else:
        ferr = np.full_like(f, np.nan)

    # Nos quedamos solo con puntos finitos tanto en tiempo como en flujo
    mask = np.isfinite(t) & np.isfinite(f)
    t, f, ferr = t[mask], f[mask], ferr[mask]

    if len(t) < 10:
        raise RuntimeError("Hay muy pocos puntos válidos en la curva de luz para analizar.")

    # Ordenamos cronológicamente por si vienen desordenados al unir piezas
    order = np.argsort(t)
    return t[order], f[order], ferr[order]


def compute_rotational_periodogram(time_days, flux, flux_err=None,
                                   min_period_h=0.5, max_period_d=15.0,
                                   samples_per_peak=10):
    """
    Calcula un periodograma de Lomb-Scargle para buscar periodos rotacionales.
    A diferencia de BLS (pensado para tránsitos tipo caja), Lomb-Scargle va mejor
    con modulaciones rotacionales suaves/cuasi-sinusoidales de cuerpos irregulares.
    """
    t = np.asarray(time_days, dtype=float)
    f = np.asarray(flux, dtype=float)

    # Cadencia mediana en días y ventana total observada
    dt_med = float(np.nanmedian(np.diff(t)))
    total_span = float(t.max() - t.min())

    # Acotamos periodos mínimos y máximos según el muestreo y la duración total
    min_p = max(min_period_h / 24.0, 2.0 * dt_med)
    max_p = min(max_period_d, max(min_p * 2.0, total_span / 2.0))

    # Construimos la malla de frecuencias con muestreo suficiente por pico
    min_freq = 1.0 / max_p
    max_freq = 1.0 / min_p
    df = 1.0 / (samples_per_peak * max(total_span, 1e-6))
    freqs = np.arange(min_freq, max_freq, df)

    if len(freqs) < 10:
        freqs = np.linspace(min_freq, max_freq, 500)

    # Armamos una LightCurve temporal para usar el método lombscargle de lightkurve
    if flux_err is not None and np.any(np.isfinite(flux_err)):
        valid_err = np.isfinite(flux_err) & (flux_err > 0)
        fallback_err = np.nanmedian(flux_err[valid_err]) if np.any(valid_err) else 1e-3
        err_clean = np.where(valid_err, flux_err, fallback_err)
        lc_tmp = lk.LightCurve(time=t, flux=f, flux_err=err_clean)
    else:
        lc_tmp = lk.LightCurve(time=t, flux=f)

    pg = lc_tmp.to_periodogram(method="lombscargle", frequency=freqs * (1.0 / u.d))

    best_period_d = _val(pg.period_at_max_power)
    best_power = _val(pg.max_power)

    # Guardamos un resumen rápido del resultado y de los datos
    summary = {
        "best_period_d": best_period_d,
        "best_period_h": best_period_d * 24.0,
        "best_power": best_power,
        "cadence_min": dt_med * 24.0 * 60.0,
        "span_d": total_span,
        "n_points": int(len(t)),
    }
    return pg, summary


def fold_rotational_curve(time_days, flux, period_days, t0=None):
    """
    Pliega la curva de luz en fase rotacional (valores entre -0.5 y 0.5 ciclos)
    y calcula una estimación sencilla de amplitud (rango entre percentiles 5 y 95)
    para evitar que algún punto atípico dispare la amplitud.
    """
    if period_days <= 0 or not np.isfinite(period_days):
        raise ValueError("El periodo debe ser un número positivo y finito.")

    t = np.asarray(time_days, dtype=float)
    f = np.asarray(flux, dtype=float)

    # Si no pasamos época de referencia, tomamos el primer tiempo
    epoch = float(t[0]) if t0 is None else float(t0)

    # Fase centrada en [-0.5, 0.5)
    phase = ((t - epoch + 0.5 * period_days) % period_days) / period_days - 0.5

    # Ordenamos por fase para que graficar o agrupar sea directo
    order = np.argsort(phase)
    phase_sorted = phase[order]
    flux_sorted = f[order]

    # Amplitud robusta usando percentiles 5% y 95%
    q05, q95 = np.nanpercentile(flux_sorted, [5.0, 95.0])
    robust_amp = float(q95 - q05)

    return phase_sorted, flux_sorted, robust_amp
