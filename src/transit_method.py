import lightkurve as lk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# 1. Parametros de BLS segun cuanto tiempo cubren los datos
# ------------------------------------------------------------
def parametros_bls(tiempo_total):
    if tiempo_total <= 30:
        # TESS - 1 sector (~27 dias)
        minimum_period = 0.3
        maximum_period = min(12.0, tiempo_total / 2)
        duraciones = np.linspace(0.01, 0.15, 10)
        factor = 10
    elif tiempo_total <= 120:
        # Kepler - 1 trimestre (~90 dias)
        minimum_period = 0.5
        maximum_period = min(40.0, tiempo_total / 2)
        duraciones = np.linspace(0.04, 0.20, 10)
        factor = 10
    else:
        # Kepler - mision completa (~4 anios)
        minimum_period = 0.5
        maximum_period = min(100.0, tiempo_total / 2)
        duraciones = np.linspace(0.03, 0.20, 12)   # 0.7 a 4.8 h
        factor = 100 

    return minimum_period, maximum_period, duraciones, factor


# ------------------------------------------------------------
# 2. Funcion principal
# ------------------------------------------------------------
def buscar(estrella, mision="Kepler"):

    
    if mision == "Kepler":
        resultados = lk.search_lightcurve(estrella, mission="Kepler",
                                          author="Kepler", cadence="long")
    else:
        resultados = lk.search_lightcurve(estrella, mission="TESS",
                                          author="SPOC", exptime=120)

    if len(resultados) == 0:
        print("No se encontraron observaciones para esta estrella.")
        return None

    #  Descargar TODOS los trimestres/sectores y pegarlos
    coleccion = resultados.download_all()
    curva_luz = coleccion.stitch()

    plt.figure()
    plt.scatter(curva_luz.time.value, curva_luz.flux.value, s=1, color="purple")
    plt.xlabel("Tiempo (dias)")
    plt.ylabel("Flujo")
    plt.title(f"{estrella} - Datos crudos")
    plt.tight_layout()
    plt.show()

    # Limpiar: quitar NaN y SOLO los picos hacia arriba

    curva_luz = curva_luz.remove_nans()
    curva_luz = curva_luz.remove_outliers(sigma_upper=4, sigma_lower=20)

    #Normalizar
    curva_luz = curva_luz.normalize()

    # Quitar tendencias
    # La ventana debe ser MAS LARGA que el transito (~1 dia).
    cadencia = np.median(np.diff(curva_luz.time.value))   # dias entre puntos
    ventana = int(1.0 / cadencia)
    if ventana % 2 == 0:       # flatten necesita un numero impar
        ventana += 1
    curva_luz = curva_luz.flatten(window_length=ventana, polyorder=2)

    #  Binning solo si los datos son muy rapidos
    if cadencia * 1440 < 10:
        curva_luz = curva_luz.bin(time_bin_size=10 / 1440)   # 10 minutos
        curva_luz = curva_luz.remove_nans()

    plt.figure()
    plt.scatter(curva_luz.time.value, curva_luz.flux.value, s=1, color="steelblue")
    plt.xlabel("Tiempo (dias)")
    plt.ylabel("Flujo normalizado")
    plt.title(f"{estrella} - Curva limpia y aplanada")
    plt.tight_layout()
    plt.show()

    # Parametros automaticos
    tiempo_total = curva_luz.time.max().value - curva_luz.time.min().value
    p_min, p_max, duraciones, factor = parametros_bls(tiempo_total)
    print(f"Datos cubren {tiempo_total:.1f} dias, "
          f"buscando periodos entre {p_min} y {p_max:.1f} dias")

    #  BLS paso 1: busqueda gruesa
    bls = curva_luz.to_periodogram(
        method="bls",
        minimum_period=p_min,
        maximum_period=p_max,
        duration=duraciones,
        frequency_factor=factor,
    )

    bls.plot()          
    plt.title(f"{estrella} - Periodograma BLS (grueso)")
    plt.show()

    periodo_grueso = bls.period_at_max_power.value
    print(f"Periodo grueso: {periodo_grueso:.4f} dias")

    # BLS paso 2: busqueda fina alrededor del pico
    periodos_finos = np.linspace(periodo_grueso * 0.995,
                                 periodo_grueso * 1.005, 3000)
    bls_fino = curva_luz.to_periodogram(
        method="bls",
        period=periodos_finos,
        duration=duraciones,
    )

    periodo = bls_fino.period_at_max_power
    t0 = bls_fino.transit_time_at_max_power
    duracion = bls_fino.duration_at_max_power
    profundidad = bls_fino.depth_at_max_power
    potencia_maxima = bls_fino.max_power
    print(f"Periodo fino: {periodo.value:.5f} dias")

    # Plegar usando tambien el tiempo del transito (t0) 
    curva_sup = curva_luz.fold(period=periodo, epoch_time=t0)

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    ejes[0].scatter(curva_sup.time.value, curva_sup.flux.value, s=2, color="pink")
    ejes[0].set_title(f"Plegada (P = {periodo.value:.5f} d)")
    ejes[0].set_xlabel("Dias desde el transito")
    ejes[0].set_ylabel("Flujo normalizado")

    # Zoom al transito
    ejes[1].scatter(curva_sup.time.value, curva_sup.flux.value, s=2, color="pink")
    ancho = 3 * duracion.value
    ejes[1].set_xlim(-ancho, ancho)
    ejes[1].set_title("Zoom al transito")
    ejes[1].set_xlabel("Dias desde el transito")
    plt.tight_layout()
    plt.show()

    #  Resumen
    tabla = pd.DataFrame({
        "estrella": [estrella],
        "periodo_dias": [float(periodo.value)],
        "duracion_transito_horas": [float(duracion.value) * 24],
        "profundidad_transito": [float(profundidad)],
        "potencia_bls": [float(potencia_maxima)],
        "num_puntos_curva": [len(curva_luz.time)],
        "flujo_promedio": [float(np.nanmean(curva_luz.flux.value))],
        "flujo_desviacion_std": [float(np.nanstd(curva_luz.flux.value))],
    }).T

    print("\nTabla resumen:")
    print(tabla)

    return curva_luz, periodo, tabla

