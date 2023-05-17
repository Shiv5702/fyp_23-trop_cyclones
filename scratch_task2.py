import math
import cv2
import netCDF4
import numpy as np
import sobel_task1
from PIL import Image
import matplotlib.pyplot as plt

"""Calculate distance between lat and lon coordinates"""
def distance(point1, point2):
    radius = 6371  # earth radius (km)
    lon1, lat1 = point1[0], point1[1]
    lon2, lat2 = point2[0], point2[1]
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

# Normalizes gradients
def normalize_gradients(gradient_vectors, width, height):
    norm_grad_vectors = []
    for k in range(len(gradient_vectors)):
        vector = []
        for j in range(len(gradient_vectors[k][-1])):
            vector.append((gradient_vectors[k][0][j] / 8, 
                            gradient_vectors[k][1][j] / 8))
        norm_grad_vectors.append(vector)

    return norm_grad_vectors

def calculate_DAV_Numpy(gradient_vectors, radial_lines):
    
    # Calculate the dot product
    dot_product = np.dot(gradient_vectors, radial_lines)
    grad_mag = np.sqrt(np.dot(gradient_vectors, gradient_vectors))
    rad_mag = np.sqrt(np.dot(radial_lines, radial_lines))
    ind = np.where(grad_mag > 0 & rad_mag > 0)

    # Calculate the deviation angle
    deviation_angle = np.arccos(dot_product[ind] / (grad_mag[ind] * rad_mag[ind]))
    deviations = np.degrees(deviation_angle)
    variance = np.var(deviations)
    
    return variance,deviations

def calculate_DAV_Efficient(gradient_vectors, img_width, img_height, ref_lat, ref_lon, rad_dist):
    deviations = []
    for y in range(len(gradient_vectors)):
        for x in range(len(gradient_vectors[y][-1])):
            # Convert gradient vector elements to numpy array for calculations
            gradient_vector = [gradient_vectors[y][0][x], gradient_vectors[y][1][x]]
            # Normalize the gradient vector
            normalized_gradient = [gradient_vector[0] / 8, gradient_vector[1] / 8]
            if normalized_gradient[0] == 0 and normalized_gradient[1] == 0:
                continue
            
            # Check if pixel point is near enough to reference point
            pix_lat = lat_max + (y/img_height)*(lat_min - lat_max)
            pix_lon = lon_min + (x/img_width)*(lon_max - lon_min)
            if distance((ref_lon, ref_lat), (pix_lon, pix_lat)) > rad_dist:
                continue
            radial_line = calculate_radial_line((ref_lon, ref_lat), (pix_lon, pix_lat))
            if radial_line[0] == 0 and radial_line[1] == 0:
                continue
            # Calculate the dot product
            dot_product = np.dot(normalized_gradient, radial_line)
            
            # Calculate the deviation angle
            deviation_angle = np.arccos(dot_product / (math.sqrt(radial_line[0]**2 + radial_line[1]**2) * math.sqrt(normalized_gradient[0]**2 + normalized_gradient[1]**2)))
            deviation_angle_deg = np.degrees(deviation_angle)
            deviations.append(deviation_angle_deg)
    
    variance = np.var(deviations)
    return variance,deviations

def calculate_radial_line(center, pixel):
    center_x, center_y = center[0], center[1]
    pixel_x, pixel_y = pixel[0], pixel[1]
    radial_line = (pixel_x - center_x, pixel_y - center_y)
    return radial_line

def numpy_coords(image):
    w, h = image.size
    coords = np.full((h*w), 0, dtype='f,f')
    for i in range(h*w):
        coords[i] = convert_to_coord(i, w, h)
    return coords

def convert_to_coord(pixel_ind, w, h):
    x = pixel_ind % w
    y = pixel_ind / w
    pix_lat = lat_max + (y/h)*(lat_min - lat_max)
    pix_lon = lon_min + (x/w)*(lon_max - lon_min)
    return (pix_lon, pix_lat)

def calculate_radial_vectors(image, lat, lon, radial_dist, coords):
    w, h = image.size
    radial_vectors = np.where(distance((lon, lat), coords) <= radial_dist, 
                              calculate_radial_line(coords, (lon, lat)), (0,0))

    return radial_vectors

# # Histogram of the dav angles
def plot_angle_histogram(angles):
     plt.hist(angles, bins=120, range=(-360, 360), edgecolor='black')
     plt.xlabel('Angles (degrees)')
     plt.ylabel('Frequency')
     plt.title('Angle Histogram')
     plt.show()
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
#image = cv2.imread('my_plot.jpg')
image = Image.open('my_plot.jpg')
image_array = np.array(image)
gradient_magnitude, gradient_direction = sobel_task1.calculate_brightness_gradient(image)

# Convert magnitude and direction into a vector
gradient_vectors = convert_to_gradient_vectors(gradient_magnitude, gradient_direction)


print("----------------------------------------------------------------------")

# With different radial distances, calculate DAV
radial_dist = 150
ref_lat, ref_lon = 20, -80
width, height = image.size 
coordinates = numpy_coords(image)
radial_lines = calculate_radial_vectors(image, ref_lat, ref_lon, radial_dist, coordinates)
norm_grad = normalize_gradients(gradient_vectors, width, height)
variance,angle_list= calculate_DAV_Numpy(norm_grad, radial_lines)
print("Length of angles", len(angle_list))
print("Variance", variance)
plot_angle_histogram(angle_list)

# Now Mapping deviation-angle variances
"""dav_array = np.zeros((width, height))
for y in range(height):
    for x in range(width):
        ref_lat = lat_max + (y/height)*(lat_min - lat_max)
        ref_lon = lon_min + (x/width)*(lon_max - lon_min)
        variance,angle_list = calculate_DAV_Efficient(gradient_vectors, width, height, 
                                                      ref_lat, ref_lon, radial_dist)
        dav_array.itemset((x, y), variance)"""
