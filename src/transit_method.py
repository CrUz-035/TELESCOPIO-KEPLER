import re
import warnings
import astropy.units as u
import lightkurve as lk
import numpy as np
import pandas as pd
from astroquery.mast import Catalogs



warnings.filterwarnings("ignore", category=lk.LightkurveWarning)

R_SUN_TO_R_EARTH = 109.076
R_SUN_TO_R_JUPITER = 9.731

TIME_SYSTEM = {"Kepler": ("BKJD", "BJD - 2454833"),
               "TESS":   ("BTJD", "BJD - 2457000")}

N_SIG_ERR = 1

FIGURE_STYLE = {"figsize": (8, 4.5), "dpi": 150, "fontsize": 11,
                "color_raw": "0.6", "color_binned": "tab:blue",
                "color_line": "tab:red", "color_transit": "tab:orange"}


def _arr(x):
    return np.asarray(getattr(x, "value", x), dtype=float)

def _val(x):
    return float(getattr(x, "value", x))

def round_to_error(value, error, n_sig=2):
    if not (np.isfinite(value) and np.isfinite(error)) or error <= 0:
        return value, error
    decimals = n_sig - 1 - int(np.floor(np.log10(error)))
    return round(value, decimals), round(error, decimals)

def _round_sig(x, n):
    if not np.isfinite(x) or x == 0:
        return x
    return round(x, n - 1 - int(np.floor(np.log10(abs(x)))))

def _pair(value, error, n_sig=None, fallback_sig=4):
    n_sig = N_SIG_ERR if n_sig is None else n_sig
    if np.isfinite(value) and np.isfinite(error) and error > 0:
        return round_to_error(value, error, n_sig)
    return _round_sig(value, fallback_sig), np.nan

def _fmt_pair(value, error):
    if np.isfinite(error) and error > 0:
        dec = max(0, N_SIG_ERR - 1 - int(np.floor(np.log10(error))))
        return f"{value:.{dec}f} ± {error:.{dec}f}"
    return str(float(value))

def _grid_step(periods, period):
    p = np.sort(_arr(periods))
    i = int(np.argmin(np.abs(p - period)))
    j = min(i + 1, len(p) - 1)
    return float(abs(p[j] - p[i])) if j != i else float(abs(p[i] - p[i - 1]))

def download_light_curve(star_name, mission, max_obs=None):
    if mission == "Kepler":
        search = lk.search_lightcurve(star_name, mission="Kepler",
                                      author="Kepler", cadence="long")
    else:
        search = lk.search_lightcurve(star_name, mission="TESS",
                                      author="SPOC", exptime=120)
    if len(search) == 0:
        raise RuntimeError(f"No hay datos de {mission} para '{star_name}'.")
    if max_obs:
        search = search[:max_obs]
    return search.download_all(flux_column="pdcsap_flux").stitch()

def clip_upper_outliers(light_curve, sigma_upper=4.0):
    return light_curve.remove_outliers(sigma_upper=sigma_upper, sigma_lower=float("inf"))


def flatten_light_curve(light_curve, max_duration, mask=None):
    dt = np.nanmedian(np.diff(_arr(light_curve.time.value)))
    window = int(max(0.5, 3 * max_duration) / dt)
    window = window if window % 2 == 1 else window + 1
    return light_curve.flatten(window_length=window, polyorder=2,
                               break_tolerance=10, mask=mask)


def bin_light_curve(light_curve, minutes):
    binned = light_curve.bin(time_bin_size=minutes  * u.min).remove_nans()
    ok = np.isfinite(_arr(binned.flux)) & np.isfinite(_arr(binned.flux_err))
    return binned[ok]

def get_bls_ranges(total_time, max_period_user=None, min_duration=0.0):
    if total_time <= 30:
        p_min, p_max = 0.3, min(12.0, total_time / 2)
        durations = np.linspace(0.02, 0.20, 12)
    elif total_time <= 120:
        p_min, p_max = 0.5, min(40.0, total_time / 2)
        durations = np.linspace(0.03, 0.25, 12)
    else:
        p_min, p_max = 0.5, min(400.0, total_time / 3)
        durations = np.linspace(0.03, 0.40, 15)

    if max_period_user is not None:
        p_max = min(p_max, max_period_user)
    durations = durations[(durations >= min_duration) & (durations < p_min * 0.9)]
    if len(durations) == 0:
        durations = np.array([max(min_duration, 0.04)])
    return p_min, p_max, durations


def run_bls(light_curve, max_period_user=None, min_duration=0.0, max_points=80000):
    total_time = _val(light_curve.time.max()) - _val(light_curve.time.min())
    p_min, p_max, durations = get_bls_ranges(total_time, max_period_user, min_duration)
    step = durations.min() / (2 * total_time)
    n = min(int(np.log(p_max / p_min) / step) + 1, max_points)
    periods = np.geomspace(p_min, p_max, n)
    return light_curve.to_periodogram(method="bls", period=periods,
                                      duration=durations, frequency_factor=1e6)


def refine_bls(light_curve, params, width=0.01, n_points=4000):
    period, duration = params["period"], params["duration_h"] / 24
    periods = np.linspace(period * (1 - width), period * (1 + width), n_points)
    durations = np.unique(np.clip(np.linspace(0.6 * duration, 1.6 * duration, 12),
                                  0.01, 0.9 * periods.min()))
    return light_curve.to_periodogram(method="bls", period=periods,
                                      duration=durations, frequency_factor=1e6)


def extract_bls_parameters(bls):
    params = {
        "period": _val(bls.period_at_max_power),
        "t0": _val(bls.transit_time_at_max_power),
        "duration_h": _val(bls.duration_at_max_power) * 24,
        "depth_bls": _val(bls.depth_at_max_power),
        "snr": float(_arr(bls.snr)[np.nanargmax(_arr(bls.power))]),
        "power": _val(bls.max_power),
        "odd_depth": np.nan, "even_depth": np.nan, "n_transits": np.nan,
    }
    try:
        stats = bls.compute_stats(period=bls.period_at_max_power,
                                  duration=bls.duration_at_max_power,
                                  transit_time=bls.transit_time_at_max_power)
        params["odd_depth"] = _val(stats["depth_odd"][0])
        params["even_depth"] = _val(stats["depth_even"][0])
        params["n_transits"] = int(np.sum(np.asarray(stats["per_transit_count"]) > 0))
    except Exception:
        pass
    return params

def fit_ephemeris(light_curve, params, n_chunks=8, n_grid=201, bin_days=0.0):
    nan = {"period_fit": np.nan, "period_err": np.nan,
           "t0_fit": np.nan, "t0_err": np.nan, "n_chunks_used": 0}
    t, f = _arr(light_curve.time.value), _arr(light_curve.flux)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    P, t0, dur = params["period"], params["t0"], params["duration_h"] / 24
    shifts = np.linspace(-dur, dur, n_grid)
    step = shifts[1] - shifts[0]

    ns, dts, sigs = [], [], []
    for idx in np.array_split(np.arange(len(t)), n_chunks):
        if len(idx) < 20:
            continue
        tc, fc = t[idx], f[idx]
        phase0 = ((tc - t0 + 0.5 * P) % P) - 0.5 * P
        out0 = np.abs(phase0) > dur
        if out0.sum() < 10:
            continue
        sigma = 1.4826 * np.nanmedian(np.abs(fc[out0] - np.nanmedian(fc[out0])))
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        chi2 = np.full(n_grid, np.inf)
        for k, s in enumerate(shifts):
            ph = ((tc - (t0 + s) + 0.5 * P) % P) - 0.5 * P
            inn = np.abs(ph) < dur / 2
            if inn.sum() < 3 or (~inn).sum() < 3:
                continue
            ssr = (np.sum((fc[inn] - fc[inn].mean()) ** 2)
                   + np.sum((fc[~inn] - fc[~inn].mean()) ** 2))
            chi2[k] = ssr / sigma ** 2
        kmin = int(np.argmin(chi2))
        if not np.isfinite(chi2[kmin]) or kmin in (0, n_grid - 1):
            continue
        good = np.where(chi2 <= chi2[kmin] + 1.0)[0]
        ns.append((np.mean(tc) - t0) / P)
        dts.append(float(np.mean(shifts[good])))
        sigs.append(max((shifts[good.max()] - shifts[good.min()]) / 2,
                        step / np.sqrt(12)))
    if len(ns) < 3:
        return nan

    ns, dts, sigs = np.array(ns), np.array(dts), np.array(sigs)
    W = 1.0 / sigs ** 2
    S, Sx, Sy = W.sum(), (W * ns).sum(), (W * dts).sum()
    Sxx, Sxy = (W * ns ** 2).sum(), (W * ns * dts).sum()
    D = S * Sxx - Sx ** 2
    b = (S * Sxy - Sx * Sy) / D
    a = (Sxx * Sy - Sx * Sxy) / D
    chi2_fit = (W * (dts - a - b * ns) ** 2).sum()
    scale = max(1.0, chi2_fit / (len(ns) - 2))
    return {"period_fit": P + b, "period_err": float(np.sqrt(S / D * scale)),
            "t0_fit": t0 + a,
            "t0_err": float(np.sqrt(Sxx / D * scale + (bin_days / np.sqrt(12)) ** 2)),
            "n_chunks_used": int(len(ns))}

def get_stellar_radius(star_name, search_radius_deg=0.01):
    try:
        name = str(star_name).strip()
        match = re.fullmatch(r"(?i)TIC[\s_-]*(\d+)", name)
        if match:
            tabla = Catalogs.query_criteria(catalog="Tic", ID=int(match.group(1)))
        else:
            tabla = Catalogs.query_object(name, radius=search_radius_deg,
                                          catalog="TIC")
        if tabla is None or len(tabla) == 0:
            return None

        if "dstArcSec" in tabla.colnames:
            tabla.sort("dstArcSec")

        radio = tabla["rad"][0]
        if np.ma.is_masked(radio):
            return None
        radio = float(radio)
        return radio if np.isfinite(radio) and radio > 0 else None
    except Exception:
        return None


def depth_from_folded(folded, duration_days):
    phase, flux = _arr(folded.time.value), _arr(folded.flux)
    inside = np.abs(phase) < 0.3 * duration_days
    outside = np.abs(phase) > 1.0 * duration_days
    if inside.sum() < 3 or outside.sum() < 3:
        return np.nan, np.nan
    depth = np.nanmedian(flux[outside]) - np.nanmedian(flux[inside])
    sigma = 1.4826 * np.nanmedian(np.abs(flux[outside] - np.nanmedian(flux[outside])))
    return float(depth), float(sigma / np.sqrt(inside.sum()))


def estimate_planet_radius(depth, depth_error, stellar_radius=None):
    k = float(np.sqrt(max(depth, 0.0)))
    out = {"rp_rstar": k,
           "rp_rstar_err": depth_error / (2 * k) if k > 0 else np.nan,
           "rp_earth": np.nan, "rp_earth_err": np.nan,
           "rp_jupiter": np.nan, "rp_jupiter_err": np.nan}
    if stellar_radius:
        out["rp_earth"] = k * stellar_radius * R_SUN_TO_R_EARTH
        out["rp_jupiter"] = k * stellar_radius * R_SUN_TO_R_JUPITER
        out["rp_earth_err"] = out["rp_rstar_err"] * stellar_radius * R_SUN_TO_R_EARTH
        out["rp_jupiter_err"] = out["rp_rstar_err"] * stellar_radius * R_SUN_TO_R_JUPITER
    return out


def bin_average(x, y, width, min_points=3):
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

def analyze_transit(star_name, mission=None, stellar_radius=None, max_obs=None,
                    bin_minutes=None, max_period=None, auto_radius=True, callback=None):
    if callback:
        callback("Argument validation...")
    if not star_name or not str(star_name).strip():
        raise ValueError("star_name no puede estar vacío")
    if mission not in (None, "Kepler", "TESS"):
        raise ValueError("mission debe ser None, 'Kepler' o 'TESS'")
    if stellar_radius is not None and stellar_radius <= 0:
        raise ValueError("stellar_radius debe ser positivo (radios solares)")

    if callback:
        callback("Searching and downloading...")
    if stellar_radius is not None:
        radius_source = "usuario"
    elif auto_radius:
        stellar_radius = get_stellar_radius(star_name)
        radius_source = "catálogo TIC" if stellar_radius else "no encontrado"
    else:
        radius_source = "no proporcionado"

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
    clean_lc = raw_lc.remove_nans().normalize()
    total_time = _val(clean_lc.time.max()) - _val(clean_lc.time.min())
    cadence_min = float(np.nanmedian(np.diff(_arr(clean_lc.time.value))) * 24 * 60)
    noise_ppm = float(np.nanmedian(_arr(clean_lc.flux_err))) * 1e6
    if bin_minutes is None:
        bin_minutes = round(2 * cadence_min) if cadence_min >= 20 else 10
    min_duration = bin_minutes / 1440
    max_duration = float(get_bls_ranges(total_time, max_period, min_duration)[2].max())

    if callback:
        callback("First step: searching for transits...")
    flat_lc = clip_upper_outliers(flatten_light_curve(clean_lc, max_duration))
    binned_lc = bin_light_curve(flat_lc, bin_minutes)
    preliminary = extract_bls_parameters(
        run_bls(binned_lc, max_period, min_duration, max_points=40000))

    if callback:
        callback("Second step: flatten, global BLS and refining...")
    mask = clean_lc.create_transit_mask(period=preliminary["period"],
                                        transit_time=preliminary["t0"],
                                        duration=2.0 * preliminary["duration_h"] / 24)
    flat_lc = clip_upper_outliers(flatten_light_curve(clean_lc, max_duration, mask=mask))
    binned_lc = bin_light_curve(flat_lc, bin_minutes)
    bls = run_bls(binned_lc, max_period, min_duration)
    params = extract_bls_parameters(bls)
    fine_bls = refine_bls(binned_lc, params)
    fine = extract_bls_parameters(fine_bls)
    if fine["power"] >= 0.9 * params["power"]:
        params = fine
        period_step = _grid_step(fine_bls.period, params["period"])
    else:
        period_step = _grid_step(bls.period, params["period"])

    ephem = fit_ephemeris(binned_lc, params, bin_days=bin_minutes / 1440)

    if callback:
        callback("Folding, depth and radius...")
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
    time_name, time_def = TIME_SYSTEM[used_mission]

    period_value = ephem["period_fit"] if np.isfinite(ephem["period_fit"]) \
        else params["period"]
    period_r, period_err_r = _pair(period_value, ephem["period_err"])
    depth_f_r, depth_f_err_r = _pair(folded_depth * 1e6, folded_err * 1e6)
    rp_k_r, _ = _pair(radius["rp_rstar"], radius["rp_rstar_err"])
    rp_e_r, rp_e_err_r = _pair(radius["rp_earth"], radius["rp_earth_err"])
    rp_j_r, rp_j_err_r = _pair(radius["rp_jupiter"], radius["rp_jupiter_err"])

    results_df = pd.DataFrame([{
        "star": star_name,
        "mission": used_mission,
        "cadence_min": float(round(cadence_min, 1)),
        "noise_ppm": _round_sig(noise_ppm, 3),
        "period_d": period_r,
        "period_err_d": period_err_r,
        "duration_h": _round_sig(params["duration_h"], 2),
        "depth_folded_ppm": depth_f_r,
        "depth_folded_err_ppm": depth_f_err_r,
        "depth_bls_ppm": _round_sig(params["depth_bls"] * 1e6, 3),
        "snr": int(round(params["snr"])) if np.isfinite(params["snr"]) else np.nan,
        "n_transits": params["n_transits"],
        "odd_even_ratio": round(odd_even, 2) if np.isfinite(odd_even) else np.nan,
        "stellar_radius_rsun": round(stellar_radius, 3) if stellar_radius else np.nan,
        "rp_rstar": rp_k_r,
        "rp_earth": rp_e_r,
        "rp_earth_err": rp_e_err_r,
        "rp_jupiter": rp_j_r,
        "rp_jupiter_err": rp_j_err_r,
    }])

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
    phase_bin_h = max(params["duration_h"] / 12, 0.05)
    bx, by, be = bin_average(phase_hours[in_window], folded_flux[in_window], phase_bin_h)
    periods = _arr(bls.period.value)

    cad = f"{cadence_min:.1f}".rstrip("0").rstrip(".")
    p_txt = _fmt_pair(period_r, period_err_r) + " d"
    head = f"{star_name} ({used_mission}), cadencia {cad} min"
    xt = f"Tiempo ({time_name} = {time_def}, días)"
    yt = "Flujo normalizado (adimensional)"
    figures = {
        "raw": {
            "title": head, "xlabel": xt, "ylabel": yt,
            "legend": {"raw": f"Cadencia original ({cad} min)"},
        },
        "clean": {
            "title": f"{head}, bins de {bin_minutes} min", "xlabel": xt, "ylabel": yt,
            "legend": {"raw": f"Cadencia original ({cad} min)",
                       "binned": f"Promedios de {bin_minutes} min",
                       "transits": "Tránsitos predichos"},
        },
        "periodogram": {
            "title": f"Periodograma BLS de {star_name} ({used_mission})",
            "xlabel": "Periodo (días)", "ylabel": "Potencia BLS (adimensional)",
            "legend": {"peak": f"P = {p_txt}"},
        },
        "folded": {
            "title": f"{head}, plegada a P = {p_txt}",
            "xlabel": "Tiempo desde el centro del tránsito (horas)", "ylabel": yt,
            "legend": {"raw": f"Datos ({cad} min)",
                       "binned": f"Promedios en fase ({phase_bin_h * 60:.0f} min)"},
        },
    }

    plot_data = {
        "star_name": star_name,
        "mission": used_mission,
        "time_raw": _arr(clean_lc.time.value), "flux_raw": _arr(clean_lc.flux),
        "time_clean": _arr(flat_lc.time.value), "flux_clean": _arr(flat_lc.flux),
        "time_binned": t_binned, "flux_binned": _arr(binned_lc.flux),
        "transit_times": transit_times,
        "periods": periods, "power": _arr(bls.power),
        "best_period": params["period"], "best_power": params["power"],
        "phase_hours": phase_hours, "folded_flux": folded_flux,
        "binned_phase_hours": bx, "binned_folded_flux": by, "binned_folded_err": be,
        "window_hours": window_hours,
        "t0": params["t0"], "duration_h": params["duration_h"],
        "depth_ppm": params["depth_bls"] * 1e6,
        "stellar_radius": stellar_radius, "radius_source": radius_source,
        "figures": figures,
        "style": FIGURE_STYLE,
        "labels": {"time": xt, "flux": yt, "period": "Periodo (días)",
                   "power": "Potencia BLS (adimensional)",
                   "phase": "Tiempo desde el centro del tránsito (horas)"},
        "time_system": time_name,
        "cadence_min": float(round(cadence_min, 1)),
        "bin_minutes": bin_minutes,
        "phase_bin_min": round(phase_bin_h * 60),
        "noise_ppm": noise_ppm,
        #esto no va en la interfaz
        "extra": {
            "period_bls_d": params["period"],
            "period_step_d": period_step,
            "t0_fit": ephem["t0_fit"], "t0_err": ephem["t0_err"],
            "n_chunks_used": ephem["n_chunks_used"],
            "rp_rstar_err": radius["rp_rstar_err"],
            "radius_source": radius_source,
        },
    }
    if callback:
        callback("Success")
    return results_df, plot_data