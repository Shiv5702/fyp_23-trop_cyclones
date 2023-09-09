import os
import re
import pandas as pd
import numpy as np
import sobel_task1
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from intensity import caluclate_intensity
from datetime import datetime

# Specify the directory and file path
excel_file_path = 'DataSources/2021_aug_ib_tracks.xlsx'

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)


directory_path = 'Images'

# Define the North Atlantic region (in degrees)
lon_min, lon_max = -120, 0
lat_min, lat_max = -5, 60
RMS = []
wind_points = []
def numpy_coords(image):
    w, h = image.size
    lat_coords = np.full((h*w), 0, dtype='d')
    lon_coords = np.full((h*w), 0, dtype='d')
    x_coords = np.full((h*w), 0, dtype='d')
    y_coords = np.full((h*w), 0, dtype='d')
    for pixel_ind in range(h*w):
        x = pixel_ind % w
        y = pixel_ind // w
        lat_coords[pixel_ind] = lat_max + (y/h)*(lat_min - lat_max)
        lon_coords[pixel_ind] = lon_min + (x/w)*(lon_max - lon_min)
        x_coords[pixel_ind] = x
        y_coords[pixel_ind] = y
    return lon_coords, lat_coords, x_coords, y_coords
def get_ib_track_data(filename):
    
    # Use regular expression to extract the date and time
    match = re.search(r'_(\d{4})(\d{2})(\d{2})(\d{2})', filename)

    if match:
        year, month, day, hour = match.groups()

        # Create a datetime object
        formatted_datetime = datetime(int(year), int(month), int(day), int(hour), 0, 0)
        
        # Convert the datetime object to a timestamp
        timestamp = formatted_datetime
    
    

    # loop thru ib_track file to find the formatted date and get the lat and lon
    search_value = timestamp

    no_val = True
    # Loop through the 'ISO_TIME' column to find the matching row
    for index, row in df.iterrows():
        if row['ISO_TIME'] == search_value:
            usa_lat = row['USA_LAT']
            usa_lon = row['USA_LON']
            wind = row['USA_WIND']
            wind_points.append(wind)
            no_val = False
            break  # Exit the loop once a match is found
    
    if no_val == True: 
        return None
    else:
        return usa_lat, usa_lon
    
def find_coordinates_around_target(target_latitude, target_longitude, lat_coords, lon_coords, radius_km):
    # Convert radius from kilometers to degrees (approximately)
    degrees_per_km = 1 / 111.32  # Rough approximation
    radius_degrees = radius_km * degrees_per_km
    # Calculate the squared distance between each coordinate and the target
    squared_distances = (lat_coords - target_latitude)**2 + (lon_coords - target_longitude)**2

    # Find the indices of coordinates within the specified radius
    within_radius_indices = np.where(squared_distances <= radius_degrees**2)

    # Extract the coordinates within the radius
    coordinates_within_radius = [(lat_coords[i], lon_coords[i]) for i in within_radius_indices[0]]

    return coordinates_within_radius

latitudes = []
longitudes = []
dav_vals = []


# Loop through all files in the directory
for filename in os.listdir(directory_path):

    # Call get_ib_track_data once
    ib_track_data = get_ib_track_data(filename)

    # Check if ib_track_data is None
    if ib_track_data is None:
        continue

    # Extract target_latitude and target_longitude
    target_latitude, target_longitude = ib_track_data
    latitudes.append(target_latitude)
    longitudes.append(target_longitude)
    
    # For each image, we call numpy coords. 
    # Join the directory and filename
    file_path = os.path.join(directory_path, filename)
    #Convert image to numpy array
    my_image = Image.open(file_path)
    lon_pts, lat_pts, x_pts, y_pts = numpy_coords(my_image)

    

    radius_km = 35

    coordinates_within_radius = find_coordinates_around_target(target_latitude, target_longitude, lat_pts, lon_pts, radius_km)

    target_points = []
    for target_latitude, target_longitude in coordinates_within_radius:
        # Find the indices of the target coordinates in lat_pts and lon_pts
        target_index = np.where((lat_pts == target_latitude) & (lon_pts == target_longitude))
    
        if len(target_index[0]) > 0:
            target_x = x_pts[target_index[0][0]]
            target_y = y_pts[target_index[0][0]]
            m = [target_x,target_y]
            target_points.append(m)
    
    

    for i in range(len(target_points)):

        folder_name = 'DAVs'

        # Split the original filename by the dot (.) to remove the file extension
        filename_parts = filename.split('.')

        # Extract the prefix (everything before the dot) and add '_DAV.npy'
        new_filename = f"{filename_parts[0]}_DAV.npy"

        # Join the folder name and the new filename with a backslash (\)
        result = f"{folder_name}\\{new_filename}"

        array = np.load(result)
        print("-------------------------------------------------")
        print(target_points)
        x = target_points[i][0]
        y = target_points[i][1]
        x = int(x)
        y = int(y)
        dav_value = array[x][y]
        dav_vals.append(dav_value)
    
original_times = [0, 3, 6, 9, 12, 15, 18, 21]  
original_values = wind_points
print("===================================================================")
print(original_times)
print("===================================================================")
print(original_values)
new_times = np.arange(0, 24, 1)  

# Perform linear interpolation
interpolated_values = np.interp(new_times, original_times, original_values)

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
print("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")
print(known_intensity)

zz = []
for i in range(len(dav_vals)-1):
    m = caluclate_intensity(dav_vals[i])
    zz.append(m)

def calculate_rms_percentage_error(known_intensity, calculated_intensity):
    # Calculate the RMS error
    rms_error = np.sqrt(np.mean((known_intensity - calculated_intensity)**2))

    # Calculate the range of known intensity values
    intensity_range = np.max(known_intensity) - np.min(known_intensity)

    # Calculate the percentage error
    percentage_error = (rms_error / intensity_range) * 100

    return percentage_error

print(zz)
calculated_intensity = np.array(zz)
print("---------------------")
print(zz)
print(calculated_intensity)
percentage_error = calculate_rms_percentage_error(known_intensity, calculated_intensity)
print(f"RMS Percentage Error: {percentage_error:.2f}%")


