import numpy as np
import math

def calculate_intensity(dav_value):
    alpha = 1859 * (10**-6)
    beta =  1437

    lower_limit = 25
    upper_limit = 165

    # exponent = np.exp(alpha * (dav_value - beta))
    z = alpha*(dav_value - beta)
    my_exp = math.exp(z)

    intensity = (upper_limit/(1 + my_exp)) + lower_limit

    return intensity

