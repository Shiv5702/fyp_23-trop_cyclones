import numpy as np
import pandas as pd
import math


def sig(dav):
    a = 1859*(10**-6)
    b = 1437
    sig = 140/(1+ math.exp(a(dav-b))) + 25
    return sig

# image = 'intensity\my_plot_ahmed.jpg'

# def numpy_coords(image):
#     w, h = image.size
#     lat_coords = np.full((h*w), 0, dtype='d')
#     lon_coords = np.full((h*w), 0, dtype='d')
#     x_coords = np.full((h*w), 0, dtype='d')
#     y_coords = np.full((h*w), 0, dtype='d')
#     for pixel_ind in range(h*w):
#         x = pixel_ind % w
#         y = pixel_ind // w
#         lat_coords[pixel_ind] = lat_max + (y/h)*(lat_min - lat_max)
#         lon_coords[pixel_ind] = lon_min + (x/w)*(lon_max - lon_min)
#         x_coords[pixel_ind] = x
#         y_coords[pixel_ind] = y
#     return lon_coords, lat_coords, x_coords, y_coords

# print(numpy_coords(image))





###############################################################################################################################################

"""
Steps 

Image --> numpy_coords

lon_pts, lat_pts, x_pts, y_pts = numpy_coords(image)

target latitude, target lon = ib tracks data 

coordinates_within_radius = find_coordinates_around_target(target_latitude, target_longitude, lat_pts, lon_pts, radius_km)

[target_lat_1 ] = coordinates_within_radius[0]
[target_lat_2 ] = coordinates_within_radius[1]



1. Find ib tracks data from list of netcdf files (The TC Centres USA lat/lon)

2. For an image, run numpy_coords returns the lon/lat and x and y coords. 


"""
# 


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
# # Loop through all files in the input folder
# for filename in os.listdir(input_folder_path):
    
#     # my_date = 




array = np.load('DAVs/merg_2021081100_DAV.npy')

print(array[159][147]) # Corresponding x point and y point 
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