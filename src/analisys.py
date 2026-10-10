import numpy as np
import pandas as pd
import lightkurve as lk

def search_star(target, mission_filter=None):
    if mission_filter:
        search_result = lk.search_lightcurve(target, mission=mission_filter)
    else:
        search_result = lk.search_lightcurve(target)

    return search_result

def get_stardata(target):

    search_result = lk.search_lightcurve(target)

    if len(search_result) == 0:
        raise ValueError(f"No result for '{target}'.")

    lc = search_result[0].download()

    mission = search_result[0].mission
    author = search_result.author

    return lc, mission, author

    