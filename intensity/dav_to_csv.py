import numpy as np
import pandas as pd
import math


def sig(dav):
    a = 1859*(10**-6)
    b = 1437
    sig = 140/(1+ math.exp(a(dav-b))) + 25
    return sig

# image = 'intensity\my_plot_ahmed.jpg'

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

# print(numpy_coords(image))





###############################################################################################################################################

"""
Steps 

Image --> numpy_coords

lon_pts, lat_pts, x_pts, y_pts = numpy_coords(image)

target latitude, target lon = ib tracks data 

coordinates_within_radius = find_coordinates_around_target(target_latitude, target_longitude, lat_pts, lon_pts, radius_km)

[target_lat_1 ] = coordinates_within_radius[0][0]
[target_lon_1 ] = coordinates_within_radius[0][1]

zzzzzzzzz = collect corresponding x point/y point from loop: for target_latitude, target_longitude in coordinates_within_radius (x,y):


dav = []
loop through zzzzzzzz:
    array = np.load("DAVs\merg_2021081123_DAV.npy")
    dav_value = print(array[x][y])
    dav.append (dav_value)


From intensity_errors = orginial_values ---> ib tracks wind 
Interpolate

RMS = []
loop thru dav:
    np.array(calculate dav[i], calculate dav[i+1])
    tt = call rms func
    RMS.append(tt)

print(RMS)
1. Find ib tracks data from list of netcdf files (The TC Centres USA lat/lon)

2. For an image, run numpy_coords returns the lon/lat and x and y coords. 

"""


import os
import numpy as np
import pandas as pd
from PIL import Image
import os
from find_coord import numpy_coords
from sobel_task1 import apply_sobel_filter
# # Specify the input folder path
# input_folder_path = 'DAVs'

# # Specify the output folder path
# output_folder_path = 'intensity\output_files'

# # Ensure the output folder exists; create it if it doesn't
# if not os.path.exists(output_folder_path):
#     os.makedirs(output_folder_path)

# # Loop through all files in the input folder
# for filename in os.listdir(input_folder_path):
#     file_path = os.path.join(input_folder_path, filename)
    
#     # Check if the item in the folder is a file and has the .npy extension
#     if os.path.isfile(file_path) and filename.endswith('.npy'):
#         # Load the .npy file into a NumPy array
#         loaded_array = np.load(file_path)
        
#         # Convert the NumPy array to a DataFrame
#         data_df = pd.DataFrame(loaded_array)
        
#         # Specify the output Excel file path in the output folder
#         output_excel_path = os.path.join(output_folder_path, os.path.splitext(filename)[0] + '.xlsx')
        
#         # Write the DataFrame to the Excel file
#         data_df.to_excel(output_excel_path, engine='openpyxl', index=False)
        
#         print(f'Data from {file_path} has been successfully written to {output_excel_path}.')


# Now finding all the coordinates 

# my_coords = []
# Loop through all files in the input folder
    
import os
import re
import pandas as pd

# Specify the directory and file path
excel_file_path = 'DataSources\ibtracs.NA.list.v04r00.csv'

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)


directory_path = 'Images'

RMS = []
wind_points = []

def get_ib_track_data(filename):
    
    # Use regular expression to extract the date and time
    match = re.search(r'_(\d{4})(\d{2})(\d{2})(\d{2})', filename)

    if match:
        year, month, day, hour = match.groups()
        formatted_datetime = f"{day}/{month}/{year} {int(hour):02d}:{int(0):02d}"
    
    

    # loop thru ib_track file to find the formatted date and get the lat and lon
    search_value = formatted_datetime  

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
    
    if no_val = True: 
        return None
    else:
        return usa_lat, usa_lon
    
latitudes = []
longitudes = []
dav_vals = []


# Loop through all files in the directory
for filename in os.listdir(directory_path):

    # get the ib tracks data 
    if not get_ib_track_data(filename):
        continue
    target_latitude, target_longitude = get_ib_track_data(filename)
    latitudes.append(target_latitude)
    longitudes.append(target_longitude)
    
    # For each image, we call numpy coords. 
    lon_pts, lat_pts, x_pts, y_pts = numpy_coords(filename)

    

    radius_km = 9

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

        dav_value = array[target_points[i][0]][target_points[i][1]]
        dav_vals.append(dav_value)
    
original_times = [0, 3, 6, 9, 12, 15, 18, 21]  
original_values = wind_p

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

zz = []
for i in range(len(dav_vals)):
    m = caluclate_intensity(dav_vals[i])
    zz.append(m)


calculated_intensity = np.array(zz)
percentage_error = calculate_rms_percentage_error(known_intensity, calculated_intensity)
print(f"RMS Percentage Error: {percentage_error:.2f}%")








        

        

    








# print(array[159][147]) # Corresponding x point and y point 
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