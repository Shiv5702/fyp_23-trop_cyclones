import numpy as np

def caluclate_intensity(dav_value):
    alpha = 1859 * 10^+6
    beta =  1437

    lower_limit = 25
    upper_limit = 140

    exponent = np.exp(alpha * (dav_value - beta))


    intensity = (upper_limit/1 + exponent) + lower_limit

    return intensity