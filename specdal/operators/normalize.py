import pandas as pd
import numpy as np
from scipy.integrate import simps
################################################################################
def normalize(series, value_norm=None, maximum=1, wave="max", interpolate="False"):
    '''
    Normalize Spectra by maximum  value or by wavelength
        wave - the wavelength to normalize. 
               max or min to normalize to the maximun or minimum.

        interpolate - interpolate wavelength to normalize if not in index
        value_norm - specify value to normalize
        maximum - value for the new normalization
    '''


    # We normalize by a given maximum
    if value_norm is not None:
        return maximum*series/value_norm, value_norm, -1
    
    if wave == "integrate":
        value_norm = simps(series)
        return maximum*series/value_norm, value_norm, wave

    # We check if we normalize to the maximum
    if wave == "max":
        wave  = series.index[series.argmax()]

    if wave == "min":
        wave  = series.index[series.argmin()]

    # We try to normalize
    try:
        if interpolate: 
            value_norm = np.interp(wave, series.index.values, series.values)
        else:
            value_norm = series[wave]
    except:
        raise Exception(f"Wavelength {wave} not in spectra. Try interpolate=True")

    return maximum*series/value_norm, value_norm, wave
