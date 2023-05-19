import math
import cv2
import netCDF4
import numpy as np
import sobel_task1
from PIL import Image
import matplotlib.pyplot as plt
import threading

"""Calculate distance between lat and lon coordinates"""
def distance(lon1, lat1, lon2, lat2):
    radius = 6371  # earth radius (km)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) * np.sin(dlat / 2) +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) * np.sin(dlon / 2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = radius * c

    return d

def convert_to_gradient_vectors(gradient_magnitude, gradient_direction, w, h):
    gradient_x = np.full((h*w), 0, dtype='d')
    gradient_y = np.full((h*w), 0, dtype='d')
    # Iterate over each pixel
    for i in range(len(gradient_magnitude)):
        for j in range(len(gradient_magnitude[i])):
            magnitude = gradient_magnitude[i,j]
            direction = gradient_direction[i,j]
            
            # Convert direction from degrees to radians
            direction_rad = np.radians(direction)
            
            # Calculate the x and y components of the gradient vector
            x_component = magnitude * np.cos(direction_rad)
            y_component = magnitude * np.sin(direction_rad)
            
            # Create the gradient vector as [x_component, y_component]
            gradient_x[i*w + j] = x_component 
            gradient_y[i*w + j] = y_component
    
    return gradient_x, gradient_y

def calculate_DAV_Numpy(gradient_x, gradient_y, radial_x, radial_y):
    # Calculate the dot product
    dot_product = gradient_x*radial_x + gradient_y*radial_y
    grad_mag = np.sqrt(gradient_x*gradient_x + gradient_y*gradient_y)
    rad_mag = np.sqrt(radial_x*radial_x + radial_y*radial_y)
    ind = np.where(grad_mag > 0)

    # Calculate the deviation angle
    deviation_angle = np.arccos(dot_product[ind] / (grad_mag[ind] * rad_mag[ind]))
    deviations = np.degrees(deviation_angle)
    angles = np.where(deviations <= 90, deviations, deviations - 180)

    # Calculate variance if there are deviation angles
    if angles.size > 0:
        variance = np.nanvar(angles)
    else:
        variance = 0
    
    return variance,angles

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

def calculate_radial_line(center_x, center_y, pixel_x, pixel_y):
    radial_line = (pixel_x - center_x, pixel_y - center_y)
    return radial_line

def numpy_coords(image):
    w, h = image.size
    lat_coords = np.full((h*w), 0, dtype='d')
    lon_coords = np.full((h*w), 0, dtype='d')
    for pixel_ind in range(h*w):
        x = pixel_ind % w
        y = pixel_ind // w
        lat_coords[pixel_ind] = lat_max + (y/h)*(lat_min - lat_max)
        lon_coords[pixel_ind] = lon_min + (x/w)*(lon_max - lon_min)
    return lon_coords, lat_coords

def convert_to_coord(pixel_ind, w, h):
    x = pixel_ind % w
    y = pixel_ind / w
    pix_lat = lat_max + (y/h)*(lat_min - lat_max)
    pix_lon = lon_min + (x/w)*(lon_max - lon_min)
    return (pix_lon, pix_lat)

def calculate_radial_vectors(lat, lon, radial_dist, lon_lst, lat_lst):
    radial_vectors_x = lon - lon_lst 
    radial_vectors_y = lat - lat_lst
    radial_ind = np.where((0 < distance(lon, lat, lon_lst, lat_lst)) & 
                          (distance(lon, lat, lon_lst, lat_lst) <= radial_dist))

    return radial_vectors_x, radial_vectors_y, radial_ind

# # Histogram of the dav angles
def plot_angle_histogram(angles):
     plt.hist(angles, bins=120, range=(-90, 90), edgecolor='black')
     plt.xlabel('Angles (degrees)')
     plt.ylabel('Frequency')
     plt.title('Angle Histogram')
     plt.show()

def get_variance(start, end):
    for pixel_ind in range(start, end):
        x = pixel_ind % width
        y = pixel_ind // width
        ref_lat = lat_max + (y/height)*(lat_min - lat_max)
        ref_lon = lon_min + (x/width)*(lon_max - lon_min)
        radial_x, radial_y, ind = calculate_radial_vectors(ref_lat, ref_lon, 
                                                radial_dist, lon_pts, lat_pts)
        variance,angle_list= calculate_DAV_Numpy(grad_x[ind], grad_y[ind], radial_x[ind], radial_y[ind])
        dav_array[y, x] = variance

# Get coordinates from netcdf4 file
nc = netCDF4.Dataset('DataSources/merg_2022020200_4km-pixel.nc4')

# Get the variable you want to plot
var = nc.variables['Tb']

# Get the latitude and longitude values
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]

# Define the North Atlantic region (in degrees)
lon_min, lon_max = -120, 0
lat_min, lat_max = -5, 60

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
width, height = image.size 
image_array = np.array(image)
gradient_magnitude, gradient_direction = sobel_task1.calculate_brightness_gradient(image)

# Convert magnitude and direction into a vector
grad_x, grad_y = convert_to_gradient_vectors(gradient_magnitude, gradient_direction,
                                             width, height)

# With different radial distances, calculate DAV
radial_dist = 150
ref_lat, ref_lon = 20, -80
width, height = image.size 
lon_pts, lat_pts = numpy_coords(image)
radial_x, radial_y, ind = calculate_radial_vectors(ref_lat, ref_lon, 
                                                   radial_dist, lon_pts, lat_pts)
variance,angle_list= calculate_DAV_Numpy(grad_x[ind], grad_y[ind], radial_x[ind], radial_y[ind])
print("Variance", variance)
plot_angle_histogram(angle_list)

# Now Mapping deviation-angle variances
dav_array = np.zeros((height, width), dtype='d')
splits = 100
split_size = (width*height) // splits
if split_size == 0:
    splits = width*height
    split_size = (width*height) // splits                                  

threads = []                                                                
for i in range(splits):                                                 
    # determine the indices of the list this thread will handle             
    start = i * split_size                                                  
    # special case on the last chunk to account for uneven splits           
    end = (width*height) if i+1 == splits else (i+1) * split_size                 
    # create the thread                                                     
    threads.append(                                                         
        threading.Thread(target=get_variance, args=(start, end)))         
    threads[-1].start()

for t in range(len(threads)):
    threads[t].join()

print(dav_array)
