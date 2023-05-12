import math
import cv2
import netCDF4
import numpy as np
from sobel_task1 import apply_sobel_filter

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

# Get coordinates from netcdf4 file
nc = netCDF4.Dataset('DataSources/merg_2022092606_4km-pixel.nc4')

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

# With different radial distances, calculate DAV
for radial_dist in range(150, 550, 50):
    pass