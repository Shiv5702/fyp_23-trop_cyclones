import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Open the NetCDF4 file
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

# Convert pixel resolution from 4km to 10km
resolution_ratio = int(10 / 4)
lon_inds = lon_inds[::resolution_ratio]
lat_inds = lat_inds[::resolution_ratio]

# Further subsample the indices to reduce the number of pixels
subsampling_ratio = 5
lon_inds = lon_inds[::subsampling_ratio]
lat_inds = lat_inds[::subsampling_ratio]

# Create a 2D meshgrid of latitudes and longitudes for the desired region
lon_subset, lat_subset = np.meshgrid(lon[lon_inds], lat[lat_inds])

# Select the variable values for the desired region
var_subset = var[0, lat_inds, lon_inds]

# Plot the variable using Matplotlib's pcolormesh function
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
im = ax.pcolormesh(lon_subset, lat_subset, var_subset, cmap='jet_r', vmin=np.min(var_subset), vmax=280)
im.set_clim(vmin=np.min(var_subset), vmax=280)
ax.set_xticks(np.arange(lon_min, lon_max+15, 10))
ax.set_yticks(np.arange(lat_min, lat_max+15, 10))
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

ax.set_xticks([])
ax.set_yticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Remove the axis and set the figure size to match the plot size
ax.axis('off')

# Save the plot
plt.savefig("my_plot.jpg", bbox_inches='tight', pad_inches=0)

# Close the NetCDF4 file
nc.close()
