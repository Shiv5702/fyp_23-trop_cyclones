import numpy as np
from scipy.interpolate import CubicSpline
from intensity import caluclate_intensity
import sobel_task1
from PIL import Image
from find_coord import *
from datetime import datetime, timedelta
import os
# from storm_ib_tracks import organize_storm_data, interpolate_and_add_variables
import pprint

# storm_directory = 'intensity/IB_Extracted'

# storms_data = organize_storm_data(storm_directory)


# # Loop through the storm data
# for storm in storms_data:
#     for date, day_data in storm['days'].items():
#         # Call the function to interpolate and add variables
#         interpolated_data = interpolate_and_add_variables(day_data)
#         # Update the day's data with the interpolated data
#         storm['days'][date] = interpolated_data



def calculate_rms_error(known_intensity, calculated_intensity):
    # Calculate the RMS error
    rms_error = np.sqrt(np.mean((known_intensity - calculated_intensity)**2))

    return rms_error




# Original data points 
original_times = [0, 3, 6, 9, 12, 15, 18, 21]  
original_values = [35,35,35,37,40,40,35,32]

latitudes = [17.3,17.5223,17.7,17.842,18,18.219,18.5,18.83]  
longitudes = [-66.1,-66.8577,-67.6,-68.3142,-69,-69.606,-70.4,-71.1238]  

new_times = np.arange(0, 24, 1)  # new_times = [0,1,2,3,4,5...23]

# Perform linear interpolation
interpolated_values = np.interp(new_times, original_times, original_values) # 24 values interpolated of WIND


# Create an array of time points for the measurements
time_points = np.arange(0, 24, 3)  # [ 0  3  6  9 12 15 18 21]


interpolation_time_points = np.arange(0, 24, 1)  # [0,1,2,3,4,5...23]

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

# interpolated_latitudes & interpolated_longitudes hold 24 values of lat/lon


# Directory containing DAV numpy files
dav_directory = "DAVs/"
image_directory = "Images/"

# Define the start date and time
start_datetime = datetime(2021, 8, 11, 0, 0)

# Define the number of hours you want to process
num_hours = 24

radius_km = 36
dav_values = []
i = 0
j =0

for hour in range(num_hours):
    #Loads the numpy array of the current hour
    # Format the hour as a string with leading zeros
    hour_str = start_datetime.strftime("%Y%m%d%H")

    # Get the entire date as a string
    datetime_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
    
    # Construct the file path for the DAV numpy array
    file_path = os.path.join(dav_directory, f"merg_{hour_str}_DAV.npy")
    
    # Load the numpy array from the file
    dav_array = np.flipud(np.load(file_path))

    # Construct the file path for the corresponding image
    image_path = os.path.join(image_directory, f"merg_{hour_str}.jpg")

    #Convert image to numpy array
    image = Image.open(image_path)
    width, height = image.size
    image_gray = image.convert('L')
    gradient_x, gradient_y = sobel_task1.apply_sobel_filter(np.array(image))
    grad_x = np.reshape(gradient_x, width * height)
    grad_y = np.reshape(gradient_y, width * height)
    lon_pts, lat_pts, x_pts, y_pts = numpy_coords(image)
        
    #finds the dav value at the tc centre coordinates
    while i< len(interpolated_latitudes):
        xy_coords = []
        target_latitude = interpolated_latitudes[i]
        target_longitude = interpolated_longitudes[j]

        coordinates_within_radius = find_coordinates_around_target(target_latitude, target_longitude, lat_pts, lon_pts, radius_km)

        for target_latitude, target_longitude in coordinates_within_radius:
        # Find the indices of the target coordinates in lat_pts and lon_pts
            target_index = np.where((lat_pts == target_latitude) & (lon_pts == target_longitude))
            
            if len(target_index[0]) > 0:
                target_x = x_pts[target_index[0][0]]
                target_y = y_pts[target_index[0][0]]
                xy_coords.append((target_x,target_y))
            else:
                print(f"Coordinates ({target_latitude}, {target_longitude}) not found in the image.")

        total = 0  
        for x,y in xy_coords:
            total = total + dav_array[int(x)][int(y)]
        
        avg_dav = total/len(xy_coords)
        dav_values.append(avg_dav)

        i = i+1
    
    # Increment the datetime by one hour
    start_datetime += timedelta(hours=1)



  
calculated_intensity = []
for dav in dav_values:
    calculated_intensity.append(caluclate_intensity(dav))

known_intensity = interpolated_values  

print(known_intensity)
print(calculated_intensity)

percentage_error = calculate_rms_error(known_intensity, calculated_intensity)
print(f"RMS  Error: {percentage_error:.2f}")
