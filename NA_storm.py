import netCDF4
import numpy as np
import matplotlib.pyplot as plt

# Open the NetCDF4 file
nc = netCDF4.Dataset('fyp_23-trop_cyclones/DataSources/merg_2022020200_4km-pixel.nc4')

# Get the variable you want to plot
var = nc.variables['Tb']

# Get the latitude and longitude values
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]

# Define the North Atlantic region (in degrees)
lon_min, lon_max = -100, 0
lat_min, lat_max = 0, 60

# Find the indices of the latitude and longitude values that correspond to the desired region
lat_inds = np.where((lat >= lat_min) & (lat <= lat_max))[0]
lon_inds = np.where((lon >= lon_min) & (lon <= lon_max))[0]

# Create a 2D meshgrid of latitudes and longitudes for the desired region
lon_subset, lat_subset = np.meshgrid(lon[lon_inds], lat[lat_inds])

# Select the variable values for the desired region
var_subset = var[0, lat_inds[0]:lat_inds[-1]+1, lon_inds[0]:lon_inds[-1]+1]

# Plot the variable using Matplotlib's pcolormesh function
plt.figure(figsize=(10, 8))
plt.pcolormesh(lon_subset, lat_subset, var_subset, cmap='jet')
plt.colorbar()
plt.title('Tb (North Atlantic)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Close the NetCDF4 file
nc.close()
