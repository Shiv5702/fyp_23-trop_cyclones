import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Open the NetCDF4 file
nc = netCDF4.Dataset('DataSources/248-Data/merg_2022090503_4km-pixel.nc4')

# Get the variable you want to plot
var = nc.variables['Tb']

# Get the latitude and longitude values
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]

# Define the North Atlantic region (in degrees)
lon_min, lon_max = -75, -50
lat_min, lat_max = 15, 30

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

# Select the variable values for the desired region
var_subset = var[0, lat_min_ind:lat_max_ind+1, lon_min_ind:lon_max_ind+1]

# Plot the variable using Matplotlib's pcolormesh function
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
im = ax.pcolormesh(lon_subset, lat_subset, var_subset, cmap='jet')
ax.set_xticks(np.arange(lon_min, lon_max+15, 10))
ax.set_yticks(np.arange(lat_min, lat_max+15, 10))
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
# plt.colorbar(im, ax=ax, shrink=0.5)
title_date = datetime.strptime('2022020200', '%Y%m%d%H')
# plt.title(f'Tb (North Atlantic) - {title_date.strftime("%Y-%m-%d %H:%M:%S")}')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')

# Remove x and y axis info
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

plt.savefig("my_plot2x.jpg")

