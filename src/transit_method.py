import warnings
import astropy.units as u
import lightkurve as lk
import numpy as np
import pandas as pd


# Silencia avisos informativos de lightkurve para no ensuciar la consola
warnings.filterwarnings("ignore", category=lk.LightkurveWarning)

# Conversión de radios solares a radios terrestres y de Júpiter
R_SUN_TO_R_EARTH = 109.076
R_SUN_TO_R_JUPITER = 9.731


def _arr(x):
    """Convierte Quantity / arreglo enmascarado a ndarray de floats."""
    return np.asarray(getattr(x, "value", x), dtype=float)


def _val(x):
    """Convierte un Quantity/escalar de astropy a float simple."""
    return float(getattr(x, "value", x))

# ---------------------------------------------------------------------------
# 1. DESCARGA Y LIMPIEZA
# ---------------------------------------------------------------------------
def download_light_curve(star_name, mission, max_obs=None):
    """
    Descarga TODOS los trimestres/sectores de la misión y los une (stitch).
    Lanza RuntimeError si no hay datos.
    """
    if mission == "Kepler":
        search = lk.search_lightcurve(star_name, mission="Kepler",
                                      author="Kepler", cadence="long")
    else:  # TESS
        search = lk.search_lightcurve(star_name, mission="TESS",
                                      author="SPOC", exptime=120)
    if len(search) == 0:
        raise RuntimeError(f"No hay datos de {mission} para '{star_name}'.")
    if max_obs:
        search = search[:max_obs]
    return search.download_all(flux_column="pdcsap_flux").stitch()

def clip_upper_outliers(light_curve, sigma_upper=4.0):
    """
    Recorta solo outliers hacia ARRIBA (flares, rayos cósmicos). Un recorte
    simétrico borraría los puntos más profundos del tránsito.
    """
    return light_curve.remove_outliers(sigma_upper=sigma_upper, sigma_lower=float("inf"))


def flatten_light_curve(light_curve, max_duration, mask=None):
    """
    Quita tendencias lentas (Savitzky-Golay). La ventana debe ser >= 3x la duración
    máxima del tránsito, o el filtro borraría el tránsito.
    `mask` (True = tránsito) excluye los tránsitos del ajuste de la tendencia.
    """
    dt = np.nanmedian(np.diff(_arr(light_curve.time.value)))
    window = int(max(0.5, 3 * max_duration) / dt)
    window = window if window % 2 == 1 else window + 1     # debe ser impar
    return light_curve.flatten(window_length=window, polyorder=2,
                               break_tolerance=10, mask=mask)


def bin_light_curve(light_curve, minutes):
    """Promedia en bins de tiempo (reduce ruido y acelera BLS)."""
    binned = light_curve.bin(time_bin_size=minutes  * u.min).remove_nans()
    ok = np.isfinite(_arr(binned.flux)) & np.isfinite(_arr(binned.flux_err))
    return binned[ok]


# ---------------------------------------------------------------------------
# 2. BOX LEAST SQUARES
# ---------------------------------------------------------------------------
def get_bls_ranges(total_time, max_period_user=None, min_duration=0.0):
    """
    Rango de períodos y duraciones (en días) según la cobertura temporal.
    Se necesitan >= 2-3 tránsitos para confirmar un período -> P_max <= T/2 (o T/3).
    """
    if total_time <= 30:            # TESS, 1 sector (~27 d)
        p_min, p_max = 0.3, min(12.0, total_time / 2)
        durations = np.linspace(0.02, 0.20, 12)
    elif total_time <= 120:         # Kepler 1 trimestre o pocos sectores TESS
        p_min, p_max = 0.5, min(40.0, total_time / 2)
        durations = np.linspace(0.03, 0.25, 12)
    else:                           # Kepler misión completa / muchos sectores
        p_min, p_max = 0.5, min(400.0, total_time / 3)
        durations = np.linspace(0.03, 0.40, 15)

    if max_period_user is not None:
        p_max = min(p_max, max_period_user)
    durations = durations[(durations >= min_duration) & (durations < p_min * 0.9)]
    if len(durations) == 0:
        durations = np.array([max(min_duration, 0.04)])
    return p_min, p_max, durations


def run_bls(light_curve, max_period_user=None, min_duration=0.0, max_points=80000):
    """BLS global: malla logarítmica de períodos (limitada a max_points por costo)."""
    total_time = _val(light_curve.time.max()) - _val(light_curve.time.min())
    p_min, p_max, durations = get_bls_ranges(total_time, max_period_user, min_duration)
    step = durations.min() / (2 * total_time)       # deriva de fase < duración/2
    n = min(int(np.log(p_max / p_min) / step) + 1, max_points)
    periods = np.geomspace(p_min, p_max, n)
    # frequency_factor alto: solo silencia un aviso; se usa nuestra malla `periods`
    return light_curve.to_periodogram(method="bls", period=periods,
                                      duration=durations, frequency_factor=1e6)


def refine_bls(light_curve, params, width=0.01, n_points=4000):
    """BLS fino (+-1%) alrededor del pico para obtener un período más preciso."""
    period, duration = params["period"], params["duration_h"] / 24
    periods = np.linspace(period * (1 - width), period * (1 + width), n_points)
    durations = np.unique(np.clip(np.linspace(0.6 * duration, 1.6 * duration, 12),
                                  0.01, 0.9 * periods.min()))
    return light_curve.to_periodogram(method="bls", period=periods,
                                      duration=durations, frequency_factor=1e6)


def extract_bls_parameters(bls):
    """Extrae período, t0, duración, profundidad, SNR y prueba impar/par del pico BLS."""
    params = {
        "period": _val(bls.period_at_max_power),               # días
        "t0": _val(bls.transit_time_at_max_power),             # BKJD / BTJD
        "duration_h": _val(bls.duration_at_max_power) * 24,    # horas
        "depth_bls": _val(bls.depth_at_max_power),             # fracción de flujo
        "snr": float(_arr(bls.snr)[np.nanargmax(_arr(bls.power))]),
        "power": _val(bls.max_power),
        "odd_depth": np.nan, "even_depth": np.nan, "n_transits": np.nan,
    }
    # Prueba impar/par: si difieren mucho, podría ser una binaria eclipsante
    try:
        stats = bls.compute_stats(period=bls.period_at_max_power,
                                  duration=bls.duration_at_max_power,
                                  transit_time=bls.transit_time_at_max_power)
        params["odd_depth"] = _val(stats["depth_odd"][0])
        params["even_depth"] = _val(stats["depth_even"][0])
        params["n_transits"] = int(np.sum(np.asarray(stats["per_transit_count"]) > 0))
    except Exception:  # puede fallar si hay muy pocos tránsitos
        pass
    return params


# ---------------------------------------------------------------------------
# 3. PROFUNDIDAD, RADIO Y UTILIDADES DE GRÁFICAS
# ---------------------------------------------------------------------------
def depth_from_folded(folded, duration_days):
    """
    Profundidad independiente: mediana fuera del tránsito menos mediana en el 60%
    central del tránsito. Devuelve (profundidad, error).
    """
    phase, flux = _arr(folded.time.value), _arr(folded.flux)
    inside = np.abs(phase) < 0.3 * duration_days
    outside = np.abs(phase) > 1.0 * duration_days
    if inside.sum() < 3 or outside.sum() < 3:
        return np.nan, np.nan
    depth = np.nanmedian(flux[outside]) - np.nanmedian(flux[inside])
    sigma = 1.4826 * np.nanmedian(np.abs(flux[outside] - np.nanmedian(flux[outside])))
    return float(depth), float(sigma / np.sqrt(inside.sum()))


def estimate_planet_radius(depth, depth_error, stellar_radius=None):
    """
    Rp/R* = sqrt(profundidad). Estimación de primer orden: ignora el oscurecimiento
    al limbo y el parámetro de impacto.
    """
    k = float(np.sqrt(max(depth, 0.0)))
    out = {"rp_rstar": k,
           "rp_rstar_err": depth_error / (2 * k) if k > 0 else np.nan,
           "rp_earth": np.nan, "rp_jupiter": np.nan}
    if stellar_radius:
        out["rp_earth"] = k * stellar_radius * R_SUN_TO_R_EARTH
        out["rp_jupiter"] = k * stellar_radius * R_SUN_TO_R_JUPITER
    return out


def bin_average(x, y, width, min_points=3):
    """Promedio de y en bins de x (para ver la forma del tránsito plegado)."""
    edges = np.arange(x.min(), x.max() + width, width)
    idx = np.digitize(x, edges)
    xc, yc, ye = [], [], []
    for i in np.unique(idx):
        sel = idx == i
        if sel.sum() >= min_points:
            xc.append(x[sel].mean())
            yc.append(y[sel].mean())
            ye.append(y[sel].std() / np.sqrt(sel.sum()))
    return np.array(xc), np.array(yc), np.array(ye)


# ---------------------------------------------------------------------------
# 4. FUNCIÓN PRINCIPAL
# ---------------------------------------------------------------------------
def analyze_transit(star_name, mission=None, stellar_radius=None, max_obs=None,
                    bin_minutes=None, max_period=None, callback=None):
    """
    Descarga y analiza cualquier estrella de Kepler o TESS.
    Devuelve (results_df, plot_data); ver la descripción al inicio del archivo.
    """
    if callback:
        callback("Argument validation...")
    # Validación de argumentos
    if not star_name or not str(star_name).strip():
        raise ValueError("star_name no puede estar vacío")
    if mission not in (None, "Kepler", "TESS"):
        raise ValueError("mission debe ser None, 'Kepler' o 'TESS'")
    if stellar_radius is not None and stellar_radius <= 0:
        raise ValueError("stellar_radius debe ser positivo (radios solares)")

    if callback:
        callback("Searching and downloading...")
    # 1. Descarga (si mission=None prueba Kepler y luego TESS)
    raw_lc, used_mission = None, None
    for candidate in ([mission] if mission else ["Kepler", "TESS"]):
        try:
            raw_lc = download_light_curve(star_name, candidate, max_obs)
            used_mission = candidate
            break
        except RuntimeError:
            continue
    if raw_lc is None:
        raise RuntimeError(f"No hay datos de {mission or 'Kepler ni TESS'} "
                           f"para '{star_name}'.")

    if callback:
        callback("Cleaning...")
    # 2. Limpieza y parámetros de la búsqueda
    clean_lc = raw_lc.remove_nans().normalize()
    total_time = _val(clean_lc.time.max()) - _val(clean_lc.time.min())
    cadence_min = np.nanmedian(np.diff(_arr(clean_lc.time.value))) * 24 * 60
    if bin_minutes is None:   # ~1 h si la cadencia es larga (Kepler), 10 min si es corta
        bin_minutes = round(2 * cadence_min) if cadence_min >= 20 else 10
    min_duration = bin_minutes / 1440
    max_duration = float(get_bls_ranges(total_time, max_period, min_duration)[2].max())

    if callback:
        callback("First step: searching for transits...")
    # 3. Pasada 1: localizar los tránsitos (aplanado sin máscara) 
    flat_lc = clip_upper_outliers(flatten_light_curve(clean_lc, max_duration))
    binned_lc = bin_light_curve(flat_lc, bin_minutes)
    preliminary = extract_bls_parameters(
        run_bls(binned_lc, max_period, min_duration, max_points=40000))
    
    if callback:
        callback("Second step: flatten, global BLS and refining...")
    # 4. Pasada 2: aplanar protegiendo los tránsitos, BLS global + refinado 
    mask = clean_lc.create_transit_mask(period=preliminary["period"],
                                        transit_time=preliminary["t0"],
                                        duration=2.0 * preliminary["duration_h"] / 24)
    flat_lc = clip_upper_outliers(flatten_light_curve(clean_lc, max_duration, mask=mask))
    binned_lc = bin_light_curve(flat_lc, bin_minutes)
    bls = run_bls(binned_lc, max_period, min_duration)      # para el periodograma
    params = extract_bls_parameters(bls)
    fine = extract_bls_parameters(refine_bls(binned_lc, params))
    if fine["power"] >= 0.9 * params["power"]:
        params = fine

    if callback:
        callback("Folding, depth and radius...")
    # 5. Plegado, profundidad y radio
    folded = flat_lc.fold(period=params["period"], epoch_time=params["t0"])
    folded_depth, folded_err = depth_from_folded(folded, params["duration_h"] / 24)
    use_folded = np.isfinite(folded_depth) and folded_depth > 0
    radius = estimate_planet_radius(folded_depth if use_folded else params["depth_bls"],
                                    folded_err if use_folded else np.nan,
                                    stellar_radius)
    odd_even = (params["odd_depth"] / params["even_depth"]
                if np.isfinite(params["even_depth"]) and params["even_depth"] > 0
                else np.nan)
    
    if callback:
        callback("Exporting results...")
    # 6. TABLA de resultados
    results_df = pd.DataFrame([{
        "star": star_name,
        "mission": used_mission,
        "period_d": params["period"],
        "t0": params["t0"],
        "duration_h": params["duration_h"],
        "depth_bls_ppm": params["depth_bls"] * 1e6,
        "depth_folded_ppm": folded_depth * 1e6,
        "snr": params["snr"],
        "n_transits": params["n_transits"],
        "odd_even_ratio": odd_even,
        **radius,
    }])

    #7. VARIABLES para las gráficas
    t_binned = _arr(binned_lc.time.value)
    n_start = int(np.floor((t_binned.min() - params["t0"]) / params["period"]))
    n_end = int(np.ceil((t_binned.max() - params["t0"]) / params["period"]))
    transit_times = params["t0"] + params["period"] * np.arange(n_start, n_end + 1)
    transit_times = transit_times[(transit_times >= t_binned.min())
                                  & (transit_times <= t_binned.max())]

    phase_hours = _arr(folded.time.value) * 24
    folded_flux = _arr(folded.flux)
    window_hours = max(3 * params["duration_h"], 3.0)
    in_window = np.abs(phase_hours) < window_hours
    bx, by, be = bin_average(phase_hours[in_window], folded_flux[in_window],
                             max(params["duration_h"] / 12, 0.05))
    periods = _arr(bls.period.value)

    plot_data = {
        "star_name": star_name,
        # Gráfica 0: curva sin aplanar
        "time_raw": _arr(clean_lc.time.value), "flux_raw": _arr(clean_lc.flux),
        # Gráfica 1: curva completa limpia + binned + tránsitos predichos
        "time_clean": _arr(flat_lc.time.value), "flux_clean": _arr(flat_lc.flux),
        "time_binned": t_binned, "flux_binned": _arr(binned_lc.flux),
        "transit_times": transit_times,
        # Gráfica 2: periodograma
        "periods": periods, "power": _arr(bls.power),
        "best_period": params["period"], "best_power": params["power"],
        # Gráfica 3: curva plegada (zoom al tránsito)
        "phase_hours": phase_hours, "folded_flux": folded_flux,
        "binned_phase_hours": bx, "binned_folded_flux": by, "binned_folded_err": be,
        "window_hours": window_hours,
        # Parámetros del tránsito (títulos y bandas)
        "t0": params["t0"], "duration_h": params["duration_h"],
        "depth_ppm": params["depth_bls"] * 1e6,
    }
    if callback:
        callback("Success")
    return results_df, plot_data
