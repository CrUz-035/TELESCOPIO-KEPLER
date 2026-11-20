import warnings
import astropy.units as u
import lightkurve as lk
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=lk.LightkurveWarning)


def _arr(x):
    return np.asarray(getattr(x, "value", x), dtype=float)


def _val(x):
    return float(getattr(x, "value", x))


def search_neo_lightcurves(target, mission=None):
    """Busca curvas de luz disponibles en MAST para un asteroide o NEO."""
    if not target or not str(target).strip():
        raise ValueError("El nombre o designación del objeto no puede estar vacío.")
    if mission:
        return lk.search_lightcurve(str(target).strip(), mission=mission)
    return lk.search_lightcurve(str(target).strip())


def prepare_neo_lightcurve(lc):
    """Limpia valores no finitos y devuelve arreglos de tiempo, flujo y error."""
    clean = lc.remove_nans()
    t = _arr(clean.time.value)
    f = _arr(clean.flux)
    ferr = _arr(clean.flux_err) if getattr(clean, "flux_err", None) is not None else np.full_like(f, np.nan)

    mask = np.isfinite(t) & np.isfinite(f)
    t, f, ferr = t[mask], f[mask], ferr[mask]
    if len(t) < 10:
        raise RuntimeError("Insuficientes puntos válidos en la curva de luz.")

    order = np.argsort(t)
    return t[order], f[order], ferr[order]


def compute_rotational_periodogram(time_days, flux, flux_err=None,
                                   min_period_h=0.5, max_period_d=15.0,
                                   samples_per_peak=10):
    """Calcula un periodograma Lomb-Scargle en días para curvas rotacionales."""
    t = np.asarray(time_days, dtype=float)
    f = np.asarray(flux, dtype=float)

    dt_med = float(np.nanmedian(np.diff(t)))
    min_p = max(min_period_h / 24.0, 2.0 * dt_med)
    total_span = float(t.max() - t.min())
    max_p = min(max_period_d, max(min_p * 2.0, total_span / 2.0))

    min_freq = 1.0 / max_p
    max_freq = 1.0 / min_p
    df = 1.0 / (samples_per_peak * max(total_span, 1e-6))
    freqs = np.arange(min_freq, max_freq, df)
    if len(freqs) < 10:
        freqs = np.linspace(min_freq, max_freq, 500)

    lc_tmp = lk.LightCurve(time=t, flux=f)
    if flux_err is not None and np.any(np.isfinite(flux_err)):
        err_clean = np.where(np.isfinite(flux_err) & (flux_err > 0), flux_err, np.nanmedian(flux_err[flux_err > 0]))
        lc_tmp = lk.LightCurve(time=t, flux=f, flux_err=err_clean)

    pg = lc_tmp.to_periodogram(method="lombscargle", frequency=freqs * (1.0 / u.d))
    best_period_d = _val(pg.period_at_max_power)
    best_power = _val(pg.max_power)

    summary = {
        "best_period_d": best_period_d,
        "best_period_h": best_period_d * 24.0,
        "best_power": best_power,
        "cadence_min": dt_med * 24.0 * 60.0,
        "span_d": total_span,
        "n_points": int(len(t)),
    }
    return pg, summary
