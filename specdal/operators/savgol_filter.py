import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
################################################################################
# vector_normalize: divide the series such that 2-norm(series) == 1
def savgol(series, window_length, polyorder, deriv=0,
                        delta=1.0, axis=-1, mode='interp', cval=0.0):
    '''
    Savitzky-Golay filter using scipy implementation
    '''
    smooth = savgol_filter(series.values, 
                        window_length, polyorder, deriv,
                        delta, axis, mode, cval)
    smooth = pd.Series(data=smooth, index=series.index)
    smooth.index.name = "wavelength"
    return smooth
