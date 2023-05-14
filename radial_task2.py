import math
import cv2
import netCDF4
import numpy as np
from sobel_task1 import apply_sobel_filter
import matplotlib.pyplot as plt

"""Calculate distance between lat and lon coordinates"""
def distance(lat1, lon1, lat2, lon2):
    radius = 6371  # earth radius (km)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

def convert_to_gradient_vectors(gradient_magnitude, gradient_direction):
    gradient_vectors = []
    
    # Iterate over each pixel
    for i in range(len(gradient_magnitude)):
        magnitude = gradient_magnitude[i]
        direction = gradient_direction[i]
        
        # Convert direction from degrees to radians
        direction_rad = np.radians(direction)
        
        # Calculate the x and y components of the gradient vector
        x_component = magnitude * np.cos(direction_rad)
        y_component = magnitude * np.sin(direction_rad)
        
        # Create the gradient vector as [x_component, y_component]
        gradient_vector = [x_component, y_component]
        gradient_vectors.append(gradient_vector)
    
    return gradient_vectors

def calculate_DAV(gradient_vectors, radial_distance):
    deviations = []

    for k in range(len(gradient_vectors)):
        for j in range(len(gradient_vectors[k][-1])):
            # Convert gradient vector elements to numpy array for calculations
            if len(gradient_vectors[k][j]) == 0:
                continue
            gradient_vector = [gradient_vectors[k][j][0], gradient_vectors[k][j][1]]
            # Normalize the gradient vector
            #normalized_gradient = gradient_vector / np.linalg.norm(gradient_vector)
            normalized_gradient = [gradient_vector[0] / 8, gradient_vector[1] / 8]

            
            # Calculate the dot product
            dot_product = np.dot(normalized_gradient, radial_distance)
            
            # Check if all elements of the dot product array are within the valid range
            #if np.all((-1 <= dot_product) & (dot_product <= 1)):
            # Calculate the deviation angle
            deviation_angle = np.arctan(dot_product)
            deviations.append(deviation_angle)
    
    # Calculate the variance of the deviation angles if there are enough valid angles

    variance = np.var(deviations)

    
    return variance,deviations


def calculate_radial_line(center_x, center_y, pixel_x, pixel_y):
    radial_line = [pixel_x - center_x, pixel_y - center_y]
    return radial_line

# # Histogram of the dav angles
# def plot_angle_histogram(angles):
#     plt.hist(angles, bins=30, range=(0, 360), edgecolor='black')
#     plt.xlabel('Angles (degrees)')
#     plt.ylabel('Frequency')
#     plt.title('Angle Histogram')
#     plt.grid(True)
#     plt.show()
# Get coordinates from netcdf4 file
nc = netCDF4.Dataset('DataSources/merg_2022020200_4km-pixel.nc4')

# Get the variable you want to plot
var = nc.variables['Tb']

# Get the latitude and longitude values
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]

# Define the North Atlantic region (in degrees)
lon_min, lon_max = -90, -70
lat_min, lat_max = 15, 22

# Find the indices of the latitude and longitude values that correspond to the desired region
lat_inds = np.where((lat >= lat_min) & (lat <= lat_max))[0]
lon_inds = np.where((lon >= lon_min) & (lon <= lon_max))[0]

# Find the nearest index of the minimum and maximum values
lat_min_ind = lat_inds[0]
lat_max_ind = lat_inds[-1]
lon_min_ind = lon_inds[0]
lon_max_ind = lon_inds[-1]

# Create a 2D meshgrid of latitudes and longitudes for the desired region
lon_subset, lat_subset = np.meshgrid(lon[lon_min_ind:lon_max_ind+1], lat[lat_min_ind:lat_max_ind+1])

# Load the image and convert it to a numpy array
image = cv2.imread('my_plot.jpg')
gradient_magnitude, gradient_direction = apply_sobel_filter(image)

# Convert magnitude and direction into a vector
gradient_vectors = convert_to_gradient_vectors(gradient_magnitude, gradient_direction)


# print(lat_subset)
# print(lon_subset)


print("----------------------------------------------------------------------")

# With different radial distances, calculate DAV
radial_dist = 150
# ref_lat, ref_lon = 20.52, -65.3874
# for pixel_lat in lat_subset:
#     for pixel_lon in lon_subset:    
#         if distance(ref_lat, ref_lon, pixel_lat[0], pixel_lon[0]) <= radial_dist:
#             pass





variance,angle_list= calculate_DAV(gradient_vectors, 300)


print("Length of angles", len(angle_list))
print("First element", angle_list[0])


# plot_angle_histogram(angle_list)



# Now Mapping deviation-angle variances 


