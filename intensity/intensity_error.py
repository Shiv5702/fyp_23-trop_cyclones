import numpy as np
from scipy.interpolate import CubicSpline
from intensity import caluclate_intensity

# Original data points (time vs. value)
original_times = [0, 3, 6, 9, 12, 15, 18, 21]  # Hours
original_values = [35,35,35,37,40,40,35,32]

# New time points (every 30 minutes)
new_times = np.arange(0, 24.5, 1)  # 0.5 represents 30 minutes

# Perform linear interpolation
interpolated_values = np.interp(new_times, original_times, original_values)




# Define the known coordinates at 0:00 and 3:00
latitudes = [17.3,17.5223,17.7,17.842,18,18.219,18.5,18.83]  # Replace with your actual values
longitudes = [-66.1,-66.8577,-67.6,-68.3142,-69,-69.606,-70.4,-71.1238]  # Replace with your actual values

# Create an array of time points for the measurements
time_points = np.arange(0, 24, 3)  # Every 3 hours

# Create an array of time points for interpolation in 30-minute increments
interpolation_time_points = np.arange(0, 24.5, 1)  # 0.5 represents 30 minutes

# Initialize empty lists for interpolated coordinates
interpolated_latitudes = []
interpolated_longitudes = []

# Create cubic spline functions for latitude and longitude
spline_lat = CubicSpline(time_points, latitudes, bc_type='natural')
spline_lon = CubicSpline(time_points, longitudes, bc_type='natural')

# Interpolate coordinates for each interpolation_time_point
for t in interpolation_time_points:
    # Evaluate the spline functions to get interpolated values
    interpolated_lat = spline_lat(t)
    interpolated_lon = spline_lon(t)
    
    # Append the interpolated values to the lists
    interpolated_latitudes.append(interpolated_lat)
    interpolated_longitudes.append(interpolated_lon)

# Print or use the interpolated coordinates
# for t, lat, lon in zip(interpolation_time_points, interpolated_latitudes, interpolated_longitudes):
#     print(f"Time: {t:.2f} Latitude: {lat:.4f} Longitude: {lon:.4f}")


print(caluclate_intensity(2724.495727019521))