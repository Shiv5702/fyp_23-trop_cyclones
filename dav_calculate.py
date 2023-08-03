import math
import dav_map
import netCDF4
import numpy as np
import sobel_task1
from PIL import Image
import matplotlib.pyplot as plt
import threading

"""Calculate distance between lat and lon coordinates with harvesine"""
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

"""Calculate distance between lat and lon coordinates without harvesine"""
def dist_noHarv(lon1, lat1, lon2, lat2):
    c = 111.325
    d = ((np.abs(lon1 - lon2) * c)**2) + ((np.abs(lat1 - lat2) * (1/np.cos(1)) * c)**2)
    return np.sqrt(d)

def calculate_DAV_Numpy(gradient_x, gradient_y, radial_x, radial_y):
    # Calculate the dot product
    dot_product = gradient_x*radial_x + gradient_y*radial_y
    grad_mag = np.sqrt(gradient_x*gradient_x + gradient_y*gradient_y)
    rad_mag = np.sqrt(radial_x*radial_x + radial_y*radial_y)
    ind = np.where(grad_mag > 0)

    # Clip the ratios to be in range between -1 and 1
    ratios = dot_product[ind] / (grad_mag[ind] * rad_mag[ind])
    ratios = np.where(ratios >= -1, ratios, -1)
    ratios = np.where(ratios <= 1, ratios, 1)

    # Calculate the deviation angle
    deviation_angle = np.arccos(ratios)
    deviations = np.degrees(deviation_angle)
    angles = np.where(deviations <= 90, deviations, deviations - 180)

    # Calculate variance if there are deviation angles
    if angles.size > 0:
        variance = np.nanvar(angles)
    else:
        variance = 0
    
    return variance,angles

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

def calculate_radial_vectors(lat, lon, radial_dist, lon_lst, lat_lst, x_lst, y_lst):
    ref_x = ((lon - lon_min)/(lon_max - lon_min))*width
    ref_y = ((lat - lat_max)/(lat_min - lat_max))*height

    # Calculate radial vectors using image pixel coordinates
    radial_vectors_x = ref_x - x_lst 
    radial_vectors_y = y_lst - ref_y
    radial_ind = np.where((0 < distance(lon, lat, lon_lst, lat_lst)) & 
                          (distance(lon, lat, lon_lst, lat_lst) <= radial_dist))

    return radial_vectors_x, radial_vectors_y, radial_ind

# Histogram of the dav angles
def plot_angle_histogram(angles):
     plt.hist(angles, bins=120, range=(-90, 90), edgecolor='black')
     plt.xlabel('Angles (degrees)')
     plt.ylabel('Frequency')
     plt.title('Angle Histogram')
     plt.show()

# This is where the thread can make more threads to split the pixel workload
def split_pixel_indices(start, end):
    splits = 10
    split_size = (end - start) // splits
    if split_size == 0:
        splits = end - start
        split_size = (end - start) // splits                                  

    threads = []                                                                
    for i in range(splits):                                                 
        # determine the indices of the list this thread will handle             
        start_ind = i * split_size + start                                                 
        # special case on the last chunk to account for uneven splits           
        end_ind = end if i+1 == splits else ((i+1) * split_size + start)           
        # create the thread                                                     
        threads.append(                                                         
            threading.Thread(target=get_variance, args=(start_ind, end_ind)))         
        threads[-1].start()

    for t in range(len(threads)):
        threads[t].join()

def get_variance(start, end):
    for pixel_ind in range(start, end):
        x = pixel_ind % width
        y = pixel_ind // width
        ref_lat = lat_max + (y/height)*(lat_min - lat_max)
        ref_lon = lon_min + (x/width)*(lon_max - lon_min)
        radial_x, radial_y, ind = calculate_radial_vectors(ref_lat, ref_lon, 
                                                radial_dist, lon_pts, lat_pts, x_pts, y_pts)
        variance,angle_list= calculate_DAV_Numpy(grad_x[ind], grad_y[ind], 
                                                 radial_x[ind], radial_y[ind])
        dav_array[y, x] = variance

# Get coordinates from netcdf4 file
nc = netCDF4.Dataset('DataSources/merg_2022020200_4km-pixel.nc4')

# Get the variable you want to plot
var = nc.variables['Tb']

# Get the latitude and longitude values
lat = nc.variables['lat'][::2]
lon = nc.variables['lon'][::2]

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
var_subset = var[0, lat_min_ind:lat_max_ind+1, lon_min_ind:lon_max_ind+1]

# Load the image and convert it to a numpy array
image = Image.open('my_plot.jpg')
width, height = image.size 
image_gray = np.array(image.convert('L'))
gradient_x, gradient_y = sobel_task1.sobel_with_cv2(image)
grad_x = np.reshape(gradient_x, width * height)
grad_y = np.reshape(gradient_y, width * height)


# Plotting the gradient vectors
def plot_gradient_vectors_on_image(image, gradient_x, gradient_y, w, h, scale=0.01, arrow_width=0.1):
    plt.imshow(image, cmap='gray')
    for i in range(len(gradient_x)):
        x = i % w
        y = i // w
        dx = gradient_x[i] * scale
        dy = gradient_y[i] * scale
        plt.arrow(x, y, dx, dy, width=arrow_width, color='red')
    plt.axis('off')

# Plotting the grad vectors here 
plot_gradient_vectors_on_image(image, grad_x, grad_y, width, height)


# With different radial distances, calculate DAV
radial_dist = 150
ref_lat, ref_lon = 20, -80
width, height = image.size 
min_temp = np.min(var_subset)
max_temp = 280
temp_threshold = 270
lon_pts, lat_pts, x_pts, y_pts = numpy_coords(image)
radial_x, radial_y, ind = calculate_radial_vectors(ref_lat, ref_lon, 
                                                   radial_dist, lon_pts, lat_pts, x_pts, y_pts)
variance,angle_list= calculate_DAV_Numpy(grad_x[ind], grad_y[ind], 
                                         radial_x[ind], radial_y[ind])
print("Variance", variance)
print("Image size", image.size)
plot_angle_histogram(angle_list)

# Now Mapping deviation-angle variances
dav_array = np.zeros((height, width), dtype='d')
splits = 1000
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
        threading.Thread(target=split_pixel_indices, args=(start, end)))         
    threads[-1].start()

for t in range(len(threads)):
    threads[t].join()
print("DAV calculations have finished")

# get the dav values masked out where brightness temperature is too high
final_DAVs = np.where(((image_gray / 255) * (max_temp - min_temp) + min_temp) <= temp_threshold, dav_array, 0)
dav_map.generate_deviation_angle_variance_map(final_DAVs)

# ------------------------------------------------

