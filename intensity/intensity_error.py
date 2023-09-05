import numpy as np
from scipy.interpolate import CubicSpline
from intensity import caluclate_intensity

def calculate_rms_percentage_error(known_intensity, calculated_intensity):
    # Calculate the RMS error
    rms_error = np.sqrt(np.mean((known_intensity - calculated_intensity)**2))

    # Calculate the range of known intensity values
    intensity_range = np.max(known_intensity) - np.min(known_intensity)

    # Calculate the percentage error
    percentage_error = (rms_error / intensity_range) * 100

    return percentage_error



# Original data points 
original_times = [0, 3, 6, 9, 12, 15, 18, 21]  
original_values = [35,35,35,37,40,40,35,32]

new_times = np.arange(0, 24, 1)  

# Perform linear interpolation
interpolated_values = np.interp(new_times, original_times, original_values)



latitudes = [17.3,17.5223,17.7,17.842,18,18.219,18.5,18.83]  
longitudes = [-66.1,-66.8577,-67.6,-68.3142,-69,-69.606,-70.4,-71.1238]  

# Create an array of time points for the measurements
time_points = np.arange(0, 24, 3)  


interpolation_time_points = np.arange(0, 24, 1)  

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





known_intensity = interpolated_values  

calculated_intensity = np.array([caluclate_intensity(2736.0669875508693),caluclate_intensity(2728.7006328516945),caluclate_intensity(2675.2767177644314),caluclate_intensity(2659.133647506552),caluclate_intensity(2664.1007748791644),
                                 caluclate_intensity(2716.930570141763),caluclate_intensity(2761.903052609536),caluclate_intensity(2733.750818661922),caluclate_intensity(2714.371096425445),caluclate_intensity(2801.6782908503665),
                                 caluclate_intensity(2750.6164533955516),caluclate_intensity(2674.1582987809),caluclate_intensity(2704.131003920646),caluclate_intensity(2696.9023951482154),caluclate_intensity(2710.5410761496187),
                                 caluclate_intensity(2722.3439506321533),caluclate_intensity(2703.6549973875603),caluclate_intensity(2740.3192248250316),caluclate_intensity(2756.3545753574404),caluclate_intensity(2716.6674836840625),
                                 caluclate_intensity(2706.021773920027),caluclate_intensity(2723.391146845953),caluclate_intensity(2681.2136767018815),caluclate_intensity(2694.143854041722)]) 


print(known_intensity)
print(calculated_intensity)

percentage_error = calculate_rms_percentage_error(known_intensity, calculated_intensity)
print(f"RMS Percentage Error: {percentage_error:.2f}%")
