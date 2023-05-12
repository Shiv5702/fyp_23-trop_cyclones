import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Open the NetCDF4 file

#this is a big file so cant put in the git, so you need to download locally.
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

# Select the variable values for the desired region
var_subset = var[0, lat_min_ind:lat_max_ind+1, lon_min_ind:lon_max_ind+1]

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
