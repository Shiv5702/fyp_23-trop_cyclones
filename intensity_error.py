import numpy as np
from scipy.interpolate import CubicSpline
from intensity import caluclate_intensity

# Original data points (time vs. value)
original_times = [0, 3, 6, 9, 12, 15, 18, 21]  # Hours
original_values = [30, 32, 35, 35,35,35,35,35]

# New time points (every 30 minutes)
new_times = np.arange(0, 24.5, 0.5)  # 0.5 represents 30 minutes

# Perform linear interpolation
interpolated_values = np.interp(new_times, original_times, original_values)

# Print the interpolated values
# for time, value in zip(new_times, interpolated_values):
#     print(f"{time:.2f}: {value}")


print(caluclate_intensity(10))