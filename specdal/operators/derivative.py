import pandas as pd
import numpy as np

################################################################################
# derivative: calculate derivative of a spectrum
def derivative(series):
    '''
    Calculate the spectral derivative.
    '''
    return pd.Series(np.gradient(series), index=series.index)
